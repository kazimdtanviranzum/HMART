"""
Trained neural assignment policy for HMART (real policy-gradient training, NumPy).

A shared-weight attention scorer embeds each candidate request together with global
queue context, produces a logit per candidate, and a softmax selects the request an
idle robot serves. The network is trained by REINFORCE with reward-to-go and a
batch-normalized baseline, using the discrete-event simulator as the environment.
Routing / batching / MPC use the same modules as the heuristic HMART policy, so the
LEARNED component is the cooperative assignment decoder of Section 4.
"""
import numpy as np, math, time, json
from sim import Hospital, generate_requests, simulate, cvar, CLASSES

D = 10   # candidate+global feature dim
Hd = 16  # hidden width
GAMMA = 0.0

def feat(req, robot, hosp, t, pend_n, frac_stat):
    ttp = hosp.travel(robot.pos, req.p)
    proc = ttp + hosp.travel(req.p, req.d)
    slack = req.deadline - t - proc
    urg = req.w * math.exp(-max(slack, 0.0) / 20.0)
    c = [1.0 if req.cls == "STAT" else 0.0,
         1.0 if req.cls == "URGENT" else 0.0,
         1.0 if req.cls == "ROUTINE" else 0.0]
    return np.array([ttp/50.0, proc/50.0, slack/60.0, urg/8.0, req.w/8.0,
                     c[0], c[1], c[2], pend_n/50.0, frac_stat], dtype=np.float64)

class Policy:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.3, (Hd, D))
        self.b = np.zeros(Hd)
        self.u = rng.normal(0, 0.3, Hd)
        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.tstep = 0

    def params(self):
        return {"W": self.W, "b": self.b, "u": self.u}

    def forward(self, X):
        # X: (n, D) -> logits (n,)
        Z = X @ self.W.T + self.b            # (n, Hd)
        A = np.tanh(Z)
        logits = A @ self.u                  # (n,)
        return logits, A

    def probs(self, X):
        logits, A = self.forward(X)
        logits -= logits.max()
        e = np.exp(logits)
        p = e / e.sum()
        return p, A

    def grad_logprob(self, X, A, p, chosen):
        # returns dict of d(logprob_chosen)/dparam
        n = X.shape[0]
        dL = -p.copy(); dL[chosen] += 1.0        # d logprob / d logits  (n,)
        gu = (dL[:, None] * A).sum(0)            # (Hd,)
        dA = np.outer(dL, self.u)                # (n, Hd)
        dZ = dA * (1 - A**2)                      # tanh'
        gW = dZ.T @ X                             # (Hd, D)
        gb = dZ.sum(0)                            # (Hd,)
        return {"W": gW, "b": gb, "u": gu}

    def adam_step(self, grads, lr=5e-3):
        self.tstep += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self.m[k] = b1*self.m[k] + (1-b1)*g
            self.v[k] = b2*self.v[k] + (1-b2)*(g*g)
            mhat = self.m[k]/(1-b1**self.tstep)
            vhat = self.v[k]/(1-b2**self.tstep)
            getattr(self, k)[...] += lr * mhat/(np.sqrt(vhat)+eps)  # ascent

class Trainer:
    def __init__(self, policy, train=True, cvar_beta=None):
        self.pi = policy
        self.train = train
        self.cvar_beta = cvar_beta
        self.reset_episode()

    def reset_episode(self):
        self.decisions = []  # list of dict(X,A,p,chosen, reward)

    def choose_fn(self, robot, pending, hosp, t):
        pend_n = len(pending)
        frac_stat = sum(1 for r in pending if r.cls == "STAT") / max(pend_n, 1)
        X = np.stack([feat(r, robot, hosp, t, pend_n, frac_stat) for r in pending])
        p, A = self.pi.probs(X)
        if self.train:
            chosen = int(np.random.choice(len(pending), p=p))
        else:
            chosen = int(np.argmax(p))
        cr = pending[chosen]
        proc = hosp.travel(robot.pos, cr.p) + hosp.travel(cr.p, cr.d)
        urg = cr.w * math.exp(-max(cr.deadline - t - proc, 0.0) / 20.0)
        if self.train:
            self.decisions.append(dict(X=X, A=A, p=p, chosen=chosen, urg=urg, reward=0.0))
        return cr

    def feedback_fn(self, lead_twt, dist):
        if self.train and self.decisions:
            d = self.decisions[-1]
            d["reward"] = d["urg"] - 0.003 * dist

    def episode_grads(self):
        # reward-to-go
        rs = [d["reward"] for d in self.decisions]
        G = np.zeros(len(rs)); acc = 0.0
        for i in reversed(range(len(rs))):
            acc = rs[i] + GAMMA * acc
            G[i] = acc
        return G

def run_episode(pi, train, hosp, reqs, K, H, mpc=True):
    tr = Trainer(pi, train=train)
    simulate(hosp, reqs, K=K, policy="HMART", H=H, seed=1,
             choose_fn=tr.choose_fn, feedback_fn=(tr.feedback_fn if train else None),
             mpc_off=not mpc)
    return tr

def train_policy(updates=120, batch_eps=8, K=8, H=240.0, lam=1.0, log="train_log.json"):
    pi = Policy(seed=0)
    curve = []; val_curve = []
    t0 = time.time()
    val_seeds = [901, 902, 903, 904, 905]
    def validate():
        tws = []
        for s in val_seeds:
            hosp = Hospital(seed=s); reqs = generate_requests(hosp, lam=lam, H=H, seed=s)
            tr = Trainer(pi, train=False)
            m = simulate(hosp, reqs, K=K, policy="HMART", H=H, seed=s,
                         choose_fn=tr.choose_fn, feedback_fn=None, mpc_off=False)
            tws.append(m["TWT"])
        return float(np.mean(tws))
    for upd in range(updates):
        all_dec = []; ep_returns = []
        for e in range(batch_eps):
            s = upd*batch_eps + e + 1
            hosp = Hospital(seed=s)
            reqs = generate_requests(hosp, lam=lam, H=H, seed=s)
            tr = run_episode(pi, True, hosp, reqs, K, H)
            G = tr.episode_grads()
            for d, g in zip(tr.decisions, G):
                d["G"] = g
            all_dec.extend(tr.decisions)
            ep_returns.append(sum(d["reward"] for d in tr.decisions))
        Gs = np.array([d["G"] for d in all_dec])
        adv = (Gs - Gs.mean()) / (Gs.std() + 1e-6)
        grads = {"W": np.zeros_like(pi.W), "b": np.zeros_like(pi.b), "u": np.zeros_like(pi.u)}
        for d, a in zip(all_dec, adv):
            g = pi.grad_logprob(d["X"], d["A"], d["p"], d["chosen"])
            for k in grads:
                grads[k] += a * g[k]
        for k in grads:
            grads[k] /= len(all_dec)
        pi.adam_step(grads, lr=4e-3)
        curve.append(float(np.mean(ep_returns)))
        if upd % 10 == 0 or upd == updates-1:
            vt = validate(); val_curve.append((upd, vt))
            print("upd %3d  mean return %.1f  val TWT %.0f  (%.1fs)" % (upd, np.mean(ep_returns), vt, time.time()-t0))
    json.dump({"curve": curve, "val_curve": val_curve}, open(log, "w"))
    np.savez("policy.npz", W=pi.W, b=pi.b, u=pi.u)
    return pi, curve

def evaluate(pi, K=12, lam=1.8, H=480.0, seeds=range(1, 21)):
    rows = {k: [] for k in ["TWT", "WIP", "makespan", "throughput", "dist", "SL_STAT", "SL_URG", "SL_ROU"]}
    wt = []
    for s in seeds:
        hosp = Hospital(seed=s)
        reqs = generate_requests(hosp, lam=lam, H=H, seed=s)
        tr = Trainer(pi, train=False)
        m = simulate(hosp, reqs, K=K, policy="HMART", H=H, seed=s,
                     choose_fn=tr.choose_fn, feedback_fn=None, mpc_off=False)
        rows["TWT"].append(m["TWT"]); rows["WIP"].append(m["WIP"]); rows["makespan"].append(m["makespan"])
        rows["throughput"].append(m["throughput"]); rows["dist"].append(m["dist"])
        rows["SL_STAT"].append(m["SL"]["STAT"]); rows["SL_URG"].append(m["SL"]["URGENT"]); rows["SL_ROU"].append(m["SL"]["ROUTINE"])
        wt.append(m["wtard"])
    out = {k: float(np.mean(v)) for k, v in rows.items()}
    out["cvar95"] = cvar(np.concatenate(wt), 0.95)
    return out

if __name__ == "__main__":
    pi, curve = train_policy(updates=120, batch_eps=8)
    print("\nEvaluating trained policy (HMART-RL) at K=12, lam=1.8, 20 seeds ...")
    ev = evaluate(pi)
    print("HMART-RL:", {k: round(v, 3) for k, v in ev.items()})
    json.dump(ev, open("neural_eval.json", "w"))

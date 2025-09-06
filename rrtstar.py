import math, random
import numpy as np

class RRTStar2D:
    def __init__(self, bounds, obstacles, step=0.25, radius=0.75, iters=5000):
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.obs = obstacles  # list[(cx,cy,r)]
        self.step = step
        self.radius = radius
        self.iters = iters
        self.V = []  # nodes: (x,y)
        self.P = {}  # parent: idx -> idx
        self.C = {}  # cost: idx -> float

    def _collision(self, p, q):
        # segment-circle collision
        px, py = p; qx, qy = q
        vx, vy = qx-px, qy-py
        vv = vx*vx + vy*vy + 1e-9
        ok = True
        for (cx,cy,r) in self.obs:
            t = ((cx-px)*vx + (cy-py)*vy)/vv
            t = max(0.0, min(1.0, t))
            sx, sy = px + t*vx, py + t*vy
            if (sx-cx)**2 + (sy-cy)**2 <= (r**2):
                ok = False; break
        return (not ok)  # True if collides

    def _near(self, q):
        return [i for i,p in enumerate(self.V) if (p[0]-q[0])**2+(p[1]-q[1])**2 <= self.radius**2]

    def plan(self, start, goal_center, goal_r):
        self.V = [tuple(start)]
        self.P = {0: None}
        self.C = {0: 0.0}
        gi = None

        for k in range(self.iters):
            rx = random.uniform(self.xmin, self.xmax)
            ry = random.uniform(self.ymin, self.ymax)
            # nearest
            dbest, ibest = 1e9, None
            for i,p in enumerate(self.V):
                d = (p[0]-rx)**2 + (p[1]-ry)**2
                if d < dbest: dbest, ibest = d, i
            px, py = self.V[ibest]
            theta = math.atan2(ry-py, rx-px)
            q = (px + self.step*math.cos(theta), py + self.step*math.sin(theta))
            # bound
            q = (min(max(q[0], self.xmin), self.xmax), min(max(q[1], self.ymin), self.ymax))
            # collision?
            if self._collision(self.V[ibest], q): continue
            # add
            qi = len(self.V)
            self.V.append(q)
            self.P[qi] = ibest
            self.C[qi] = self.C[ibest] + math.dist(self.V[ibest], q)
            # rewire
            for j in self._near(q):
                if self._collision(self.V[j], q): continue
                cj = self.C[j] + math.dist(self.V[j], q)
                if cj < self.C[qi]:
                    self.P[qi] = j
                    self.C[qi] = cj
            # goal check
            if (q[0]-goal_center[0])**2 + (q[1]-goal_center[1])**2 <= goal_r**2:
                gi = qi; break

        if gi is None: return None  # fail
        # recover path
        path = []
        i = gi
        while i is not None:
            path.append(self.V[i]); i = self.P[i]
        path.reverse()
        return path  # list[(x,y)]

import torch
from envs import Env

# DH Parameters for Franka Emika Panda (from official documentation)
# format: (d, theta, r, alpha)
DH_PARAMS = [
    (0.333, 0., 0., -torch.pi/2),
    (0.,    0., 0.,  torch.pi/2),
    (0.316, 0., 0.0825, torch.pi/2),
    (0.,    0., -0.0825, -torch.pi/2),
    (0.384, 0., 0., torch.pi/2),
    (0.,    0., 0.088, torch.pi/2),
    (0.107, 0., 0., 0.)
]

def dh_transform(d, theta, r, alpha):
    """Computes the transformation matrix from DH parameters."""
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta)*torch.cos(alpha),  torch.sin(theta)*torch.sin(alpha), r*torch.cos(theta)],
        [torch.sin(theta),  torch.cos(theta)*torch.cos(alpha), -torch.cos(theta)*torch.sin(alpha), r*torch.sin(theta)],
        [0,               torch.sin(alpha),                     torch.cos(alpha),                    d                  ],
        [0,               0,                                    0,                                   1                  ]
    ])

class FrankaPanda7DOF(Env):
    def __init__(self, start_pos, T, K, dt, ap_map, spec):
        super().__init__(start_pos, T, K, dt, ap_map, spec)
        self.dim = 7  # 7 degrees of freedom (joint angles)
        self.link_lengths = [0.333, 0.316, 0.384, 0.107] # For collision checking
        # Joint limits (from datasheet)
        self.q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.q_max = torch.tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
        if start_pos.is_cuda:
            self.q_min = self.q_min.to(start_pos.device)
            self.q_max = self.q_max.to(start_pos.device)

    def _get_link_positions(self, q):
        """Computes the position of each joint/link for collision checking."""
        # q: (B, 7) or (7,)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        
        B = q.shape[0]
        transforms = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(q.device)
        link_positions = torch.zeros(B, 8, 3).to(q.device) # Base + 7 joints

        for i in range(7):
            d, _, r, alpha = DH_PARAMS[i]
            T = dh_transform(d, q[:, i], r, alpha).to(q.device)
            transforms = torch.bmm(transforms, T)
            link_positions[:, i+1, :] = transforms[:, :3, 3]

        return link_positions

    def forward_kinematics(self, q):
        """Computes end-effector position from joint angles q."""
        link_positions = self._get_link_positions(q)
        return link_positions[:, -1, :] # Return final link position

    def step(self, x, u):
        # Simple integrator dynamics on joint angles, respecting limits
        next_x = x + u * self.dt
        return torch.max(torch.min(next_x, self.q_max), self.q_min)

    def check_collisions(self, traj_batch, obstacles):
        """
        traj_batch: (B, T, 7)
        obstacles: list of {'center': (3,), 'radius': float}
        Returns a boolean tensor (B, T) where True means collision.
        """
        B, T, _ = traj_batch.shape
        q_flat = traj_batch.reshape(B * T, 7)
        link_positions = self._get_link_positions(q_flat) # (B*T, 8, 3)
        link_positions = link_positions.reshape(B, T, 8, 3)

        in_collision = torch.zeros(B, T, dtype=torch.bool).to(traj_batch.device)
        for obs in obstacles:
            center = obs['center'].to(traj_batch.device)
            radius = obs['radius']
            
            # Simple sphere-based collision checking for each link segment
            for i in range(7):
                p1 = link_positions[:, :, i, :]
                p2 = link_positions[:, :, i+1, :]
                
                # Line segment to point distance check
                ap = center - p1
                ab = p2 - p1
                
                t = torch.clamp(torch.sum(ap * ab, dim=-1) / torch.sum(ab * ab, dim=-1), 0, 1)
                d = torch.linalg.norm(p1 + t.unsqueeze(-1) * ab - center, dim=-1)
                
                in_collision |= (d < radius)
        return in_collision

def ap_map_franka(ap_dict, traj_batch, env_spec):
    """
    Implements atomic propositions for the Franka environment.
    traj_batch: (B, T, 7)
    """
    B, T, _ = traj_batch.shape
    ap_vals = torch.zeros(B, T, len(ap_dict), device=traj_batch.device)

    # Calculate end-effector position for the entire batch
    q_flat = traj_batch.reshape(B * T, 7)
    ee_pos = env_spec.env.forward_kinematics(q_flat).reshape(B, T, 3)

    for i, ap_name in enumerate(ap_dict):
        if ap_name == 'goal':
            goal_center = env_spec.spec_dict['goal']['center'].to(traj_batch.device)
            goal_radius = env_spec.spec_dict['goal']['radius']
            dist_to_goal = torch.linalg.norm(ee_pos - goal_center, dim=-1)
            ap_vals[:, :, i] = (dist_to_goal < goal_radius).float()
        elif ap_name == 'avoid':
            # This is a special case handled by the collision checker
            # We assume 'avoid' means collision with workspace obstacles
            collisions = env_spec.env.check_collisions(traj_batch, env_spec.spec_dict['avoid'])
            ap_vals[:, :, i] = (~collisions).float() # Note: avoid is G(~p), so we use ~p here

    return ap_vals

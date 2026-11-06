"""HoST humanoid standing-up (fall recovery) task for mjlab.

This package ports the recovery (standing-up) framework of
OpenRobotLab/HoST -- RSS 2025, "Learning Humanoid Standing-up Control across
Diverse Postures" -- from Isaac Gym + legged_gym onto mjlab's manager-based API.

What is ported
--------------
- ``mdp/rewards.py``       -- HoST's four reward groups (task / regu / style /
  target), the multiplicative stand-up task reward and the two base-height
  stage gates.
- ``mdp/observations.py``  -- HoST's single-step observation (76 dims on G1)
  plus the action-scale channel used by the training-wheels curriculum.
- ``mdp/events.py``        -- the "diverse postures" part of the paper: prone /
  supine / side initial states.
- ``mdp/curriculums.py``   -- the action-scale training wheel.
- ``mdp/pull_force.py``    -- the vertical pull-force training wheel.
- ``mdp/terminations.py``  -- velocity-limit terminations. HoST deliberately
  does not terminate on contact, which is what lets the policy drag its
  torso / arms on the ground while getting up.

Registered tasks
----------------
``Unitree-G1-HoST-StandUp``  -- flat ground, prone/supine starts.

Deviations from HoST
--------------------
Every deviation is tagged ``NOTE:`` in the module docstrings. See
``README.md`` in this package for the full list.
"""
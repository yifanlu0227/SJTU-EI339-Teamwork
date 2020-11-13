## MPC 和 TRPO的演示视频



### MPC

MPC算法非常慢（由于我们没有GPU，在CPU上计算运行）

我们在测试的时候为了加快一个iteration的速度，optimazier使用了较小的搜索次数。

- Qube实现效果较好，使用ABC和random shooting都可以实现将杆甩上去的效果 (reward=3.1)

- Swing可以学习到一去一回的迂回战略，但是冲的不够快，返回的时候无法将杆甩上去（reward=110.3）

- BallBalancer可以维持在平面上，但是移动到中心的效果不明显。（reward=597.5）

  

建议拖动进度条观看



### TRPO

TRPO算法运行速度较快。

- Qube可以实现将杆快速甩上去。（reward=5.09）
- BallBalancer可以将球以较快的速度推向中心。（reward=703.6）
- CartpoleSwingShort-v0可以看出model有将stick向上抬，但最终会卡在某个局部最优处。（reward=9431.3）
CPLMP 实验脚本包（与 HaotianZhu 仓库 src 布局配套）
================================================

本目录文件
----------
- cplmp_runtime_patch.py      记忆池 CPLMP 运行时补丁（环境变量 CPLMP_ENABLE 等）
- train_with_cplmp_patch.py   训练入口：先 apply_cplmp_patch 再调用 LecSlides_370K/train/train.py
- eval_udop_t5_10pct_baseline.py  基线评测（ROUGE / METEOR，多进程可选）
- eval_udop_t5_10pct_cplmp.py     CPLMP 评测入口（先打补丁再调用 baseline 的 main）

环境核对（本机已测）
--------------------
Conda 环境名：zht，Python 3.10.x。

在 src/experiment_tools/cplmp 下已执行且无报错：
  python train_with_cplmp_patch.py --help
  python eval_udop_t5_10pct_baseline.py --help
  python eval_udop_t5_10pct_cplmp.py --help

依赖：需能 import LecSlides_370K/train/train.py 及 model.language_model.llava_phi（即仓库
ROOT/src 与 ROOT/src/LecSlides_370K 在 PYTHONPATH 中；训练脚本已插入路径）。

说明：完整训练/评测仍需数据 JSON、checkpoint、GPU 与 nltk/rouge-score 等依赖；上述仅验证
解释器加载与参数解析链路正常。

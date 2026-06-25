# U202215450 · 本科毕业设计实验代码

**题目**：基于深度学习的教学幻灯片摘要生成研究  
**作者**：朱昊天（U202215450）

本仓库仅包含毕业设计的**实验代码**（不含模型权重、数据清单与论文 LaTeX）。

## 目录结构

```
src/
├── LecSlides_370K/          # 幻灯片摘要主实验框架（UDOP/T5 等）
├── LLaVA-main/              # LLaVA 参考实现
└── experiment_tools/
    ├── cplmp/               # CPLMP 训练与评测补丁
    └── ops/                 # 实验辅助脚本
run_baseline_smoke.sh        # 基线冒烟测试
EXPERIMENT_SUMMARY_WIN_ABLATION.md  # 窗口消融实验摘要
```

## 快速开始

```bash
# 基线冒烟测试
bash run_baseline_smoke.sh

# CPLMP 相关脚本
ls src/experiment_tools/cplmp/
```

## 本地未纳入 Git 的内容

| 内容 | 说明 |
|------|------|
| `data/` | 数据 JSON 清单 |
| `log/` | 实验指标与日志 |
| checkpoint 权重 | 需从训练环境单独获取 |
| `template-of-thesis-main/` | 论文 LaTeX（本地保留） |

## License

本项目为本科毕业设计成果，代码仅供学术交流参考。

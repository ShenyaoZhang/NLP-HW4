# Task 3 稀有&意外AE检测（整理版）

## 目录
- `task3_improved_pipeline.py`：主流程，训练 Isolation Forest，生成结果
- `task3_drug_label_filter.py`：过滤已知AE与适应症，含MedDRA同义词匹配
- `task3_interactive_query.py`：交互查询/任意组合判定（依赖已生成CSV）
- `task3_bert_clinical_features.py`：FAERS报告临床特征&风险因子分析（可选）
- `task3_data_collector.py`：从OpenFDA抓取原始drug-event数据
- `task3_data_and_model.py`：数据预处理与模型构建
- `task3_show_results.py`：展示已生成的结果文件
- `config_task3.py`：参数配置（阈值、top_k等）

## 使用说明
1) 准备数据目录：在仓库内新建 `data/`，放入：
   - 原始对偶：`task3_oncology_drug_event_pairs.csv`
   - （可选）模型文件：`data/models/task3_if_model.joblib`、`task3_scaler.joblib`
2) 跑全流程生成结果：
   ```bash
   cd task3
   python3 task3_improved_pipeline.py
   ```
   输出：`data/task3_all_unexpected_no_cap.csv`（无截断全量），`data/task3_unexpected_anomalies.csv`（Top-K+per-drug cap）。
3) 交互查询（依赖已生成的CSV）：
   ```bash
   cd task3
   python3 task3_interactive_query.py --drug "Epcoritamab" --adverse_event "Neutropenia"
   ```
   默认优先读 `data/task3_all_unexpected_no_cap.csv`，若缺失则尝试其他文件。
4) 临床特征/风险因子分析（可选，实时调用 OpenFDA API）：
   ```bash
   cd task3
   python3 task3_bert_clinical_features.py --drug "Epcoritamab" --adverse_event "Neutropenia"
   ```

## 说明
- 不包含任何数据文件，请自行放入 `data/`。
- 已移除PRR强制保留逻辑，筛选仅依据：已知AE过滤 + 适应症过滤 + 罕见性（count < 原始数据均值3.24）。
- 未包含 `task3_bert_synonym_matcher.py` 及演示脚本。

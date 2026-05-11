━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STURNUS — Full 3-Loop Protocol
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Python   : /Users/aman/Sturnus/sturnus_env/bin/python (3.12)
  Max tok  : 10000000
  Batch    : 256 tokens
  Seed     : 42
  Clean    : 1
  Warmup   : skipped
  Warm batch: 1
  Log      : /Users/aman/Sturnus/logs/sturnus-full-protocol.log
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLEAN — Wiping state and logs                                   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[clean] Preserved /Users/aman/Sturnus/logs/k_trajectory.jsonl
[clean] Preserved /Users/aman/Sturnus/logs/expert_drift.jsonl
[clean] Preserved /Users/aman/Sturnus/logs/thermal_regression_validation.jsonl
[clean] Done
[skip] Warmup phases skipped (--skip-warmup)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LOOP 1 — training_b_full  (100% token, Timeline B, 10000000 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
======================================================================
  STURNUS — Full Fine-Tuning
======================================================================
  Target tokens:   10,000,000
  Batch size:      256 tokens
  Datasets:        ultrachat, dolly_15k, alpaca_cleaned, openorca, gsm8k, wikitext, codeparrot_clean, openhermes
  Expert pool:     100
======================================================================
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.
[boot] HuggingFace auth OK
[boot] Convolution loaded
[boot] Routing memory loaded
[boot] Gate loaded (mlx-community/Qwen2.5-0.5B-Instruct-4bit)
[boot] Central deferred (will load on first use: mlx-community/Mistral-7B-Instruct-v0.3-4bit)
[boot] MAML loaded
[boot] Available RAM: 5639 MB (real-time, gate loaded)
[boot] After Central reserve: 1639 MB for experts
[boot] Expert cap: hw_max=1, hw_min=1, avg=2
[boot] Expert LRU cache: 2 concurrent
[boot] Starting training loop...

[boot] Waiting for first sample...
[data] Initialising mixture streams: ['ultrachat', 'dolly_15k', 'alpaca_cleaned', 'openorca', 'gsm8k', 'wikitext', 'codeparrot_clean', 'openhermes']
[data] Opening stream wikitext (Salesforce/wikitext)
[data] Opening stream ultrachat (HuggingFaceH4/ultrachat_200k)
[batch 1] sample | source=ultrachat | domain=general | tokens=512
[batch 1] gate | pref=B | a_l=0 | conf=0.679 | k=6 | cluster_hit=False
[batch 1] load_start | requested=[21, 3, 14, 11, 9, 15] | x_used=7 | ram_mb=6635
[warn] Skipping unloaded experts: [14, 11, 9, 15]
[batch 1] load_done | active=[21, 3] | missing=[14, 11, 9, 15]
[batch 1] experts_done | outputs=2 | fragment_size=256
[batch 1] central_done | entropy=0.1139
[batch 1] gradients_done | experts=[21, 3]
[batch 1] save_unload_done | experts=[21, 3]
[data] Opening stream alpaca_cleaned (yahma/alpaca-cleaned)
batch=1 | loss=3.0008 | k=2 | pref=B | a_l=0 | conf=0.679 | x_next=7 | thermal=62.0 | ram_mb=3989 | tok/s=17.7 | r_i=0.0000 | domain=general | experts_used=[21, 3] | total_tokens=512
[k-trajectory] batch=1 | tokens=512 | domain=general | k=2 | k10=2.00 | k100=2.00 | conf=0.679 | cluster=False | timeline_a_rate=0.000
[thermal-regression] batch=1 | history=1 | x_next=7 | bounded=True | guard=False | thermal=62.0 | source=proxy_or_estimate
[batch 2] sample | source=alpaca_cleaned | domain=general | tokens=163
[batch 2] gate | pref=B | a_l=0 | conf=0.739 | k=5 | cluster_hit=False
[batch 2] load_start | requested=[4, 7, 9] | x_used=7 | ram_mb=2875
[warn] Skipping unloaded experts: [9]
[batch 2] load_done | active=[4, 7] | missing=[9]
[batch 2] experts_done | outputs=2 | fragment_size=81
[batch 2] central_done | entropy=0.7697
[batch 2] gradients_done | experts=[4, 7]
[batch 2] save_unload_done | experts=[4, 7]
[data] Opening stream dolly_15k (databricks/databricks-dolly-15k)
batch=2 | loss=2.5372 | k=2 | pref=B | a_l=0 | conf=0.739 | x_next=7 | thermal=62.7 | ram_mb=5174 | tok/s=10.4 | r_i=0.4987 | domain=general | experts_used=[4, 7] | total_tokens=675
[k-trajectory] batch=2 | tokens=675 | domain=general | k=2 | k10=2.00 | k100=2.00 | conf=0.739 | cluster=False | timeline_a_rate=0.000
[thermal-regression] batch=2 | history=2 | x_next=7 | bounded=True | guard=False | thermal=62.7 | source=proxy_or_estimate
[batch 3] sample | source=dolly_15k | domain=general | tokens=149
[batch 3] gate | pref=B | a_l=0 | conf=0.492 | k=10 | cluster_hit=True
[batch 3] load_start | requested=[7, 4] | x_used=7 | ram_mb=5869
[batch 3] load_done | active=[7, 4] | missing=[]
[batch 3] experts_done | outputs=2 | fragment_size=74
[batch 3] central_done | entropy=0.0475
[batch 3] gradients_done | experts=[7, 4]
[batch 3] save_unload_done | experts=[7, 4]
batch=3 | loss=1.2689 | k=2 | pref=B | a_l=0 | conf=0.492 | x_next=7 | thermal=65.4 | ram_mb=2334 | tok/s=16.0 | r_i=0.4935 | domain=general | experts_used=[7, 4] | total_tokens=824
[k-trajectory] batch=3 | tokens=824 | domain=general | k=2 | k10=2.00 | k100=2.00 | conf=0.492 | cluster=True | timeline_a_rate=0.000
[thermal-regression] batch=3 | history=3 | x_next=7 | bounded=True | guard=True | thermal=65.4 | source=proxy_or_estimate
[data] Opening stream openhermes (teknium/openhermes)
batch=4 | loss=0.5311 | k=2 | pref=B | a_l=0 | conf=0.723 | x_next=7 | thermal=66.0 | ram_mb=2141 | tok/s=17.0 | r_i=0.4937 | domain=general | experts_used=[7, 4] | total_tokens=1201
[k-trajectory] batch=4 | tokens=1201 | domain=general | k=2 | k10=2.00 | k100=2.00 | conf=0.723 | cluster=True | timeline_a_rate=0.000
[thermal-regression] batch=4 | history=4 | x_next=7 | bounded=True | guard=True | thermal=66.0 | source=proxy_or_estimate
[warn] Skipping unloaded experts: [88]
[data] Opening stream openorca (Open-Orca/OpenOrca)
batch=5 | loss=0.5413 | k=2 | pref=B | a_l=0 | conf=0.661 | x_next=7 | thermal=66.2 | ram_mb=2957 | tok/s=6.9 | r_i=0.0000 | domain=general | experts_used=[99, 84] | total_tokens=1713
[k-trajectory] batch=5 | tokens=1713 | domain=general | k=2 | k10=2.00 | k100=2.00 | conf=0.661 | cluster=False | timeline_a_rate=0.000
[thermal-regression] batch=5 | history=5 | x_next=7 | bounded=True | guard=True | thermal=66.2 | source=proxy_or_estimate
batch=6 | loss=0.0740 | k=1 | pref=A | a_l=1 | conf=0.950 | x_next=7 | thermal=66.5 | ram_mb=1997 | tok/s=7.4 | r_i=0.4807 | domain=general | experts_used=[81] | total_tokens=1911
[k-trajectory] batch=6 | tokens=1911 | domain=general | k=1 | k10=1.83 | k100=1.83 | conf=0.950 | cluster=False | timeline_a_rate=0.143
[thermal-regression] batch=6 | history=6 | x_next=7 | bounded=True | guard=True | thermal=66.5 | source=proxy_or_estimate
[data] Opening stream gsm8k (openai/gsm8k)
[data] Opening stream codeparrot_clean (codeparrot/codeparrot-clean)
[warn] Skipping unloaded experts: [96, 83, 82, 86]
batch=100 | loss=1.5566 | k=2 | pref=B | a_l=0 | conf=0.683 | x_next=7 | thermal=67.9 | ram_mb=2514 | tok/s=17.7 | r_i=0.0000 | domain=code | experts_used=[90, 77] | total_tokens=31439
[k-trajectory] batch=100 | tokens=31439 | domain=code | k=2 | k10=2.00 | k100=2.00 | conf=0.683 | cluster=False | timeline_a_rate=0.485
[thermal-regression] batch=100 | history=11 | x_next=7 | bounded=True | guard=True | thermal=67.9 | source=proxy_or_estimate
[checkpoint] Saved at batch 100, 31439 tokens
[warn] Skipping unloaded experts: [22]
[warn] Skipping unloaded experts: [18, 9, 5]
[warn] Skipping unloaded experts: [80, 75, 98]
[warn] Skipping unloaded experts: [11, 20, 14, 5]
[warn] Skipping unloaded experts: [94, 98, 85]
[warn] Skipping unloaded experts: [89]
[warn] Skipping unloaded experts: [75]
[warn] Skipping unloaded experts: [85]
[warn] Skipping unloaded experts: [81, 87]
[checkpoint] Saved at batch 200, 63313 tokens
[warn] Skipping unloaded experts: [81]
[warn] Skipping unloaded experts: [93]
[warn] Skipping unloaded experts: [23, 6]
[warn] Skipping unloaded experts: [81, 87]
[warn] Skipping unloaded experts: [6]
[warn] Skipping unloaded experts: [22, 9]
[expert-drift] tokens=106845 | batch=334 | drifted=5 | top=20:general->code(0.0100), 23:general->code(0.0100), 89:general->code(0.0052), 13:general->code(0.0050), 16:general->code(0.0050)
[warn] Skipping unloaded experts: [89]
[warn] Skipping unloaded experts: [86]
[checkpoint] Saved at batch 400, 127712 tokens
[warn] Skipping unloaded experts: [75]
[warn] Skipping unloaded experts: [76, 80]
[warn] Skipping unloaded experts: [22, 9]
[warn] Skipping unloaded experts: [7, 4]
[warn] Skipping unloaded experts: [9, 6]
[warn] Skipping unloaded experts: [86]
[warn] Skipping unloaded experts: [96, 86]
[warn] Skipping unloaded experts: [76, 92]
[warn] Skipping unloaded experts: [84, 82]
[expert-drift] tokens=200983 | batch=634 | drifted=8 | top=20:general->code(0.0099), 23:general->code(0.0099), 89:general->code(0.0051), 87:code->general(0.0050), 13:general->code(0.0050)
[warn] Skipping unloaded experts: [84]
[warn] Skipping unloaded experts: [83, 76, 84]
[warn] Skipping unloaded experts: [22, 3]
[warn] Skipping unloaded experts: [86, 98]
[warn] Skipping unloaded experts: [89, 81, 84]
[warn] Skipping unloaded experts: [76]
[warn] Skipping unloaded experts: [6]
[warn] Skipping unloaded experts: [98, 80]
[warn] Skipping unloaded experts: [86, 78, 84]
[warn] Skipping unloaded experts: [81, 88]
[warn] Skipping unloaded experts: [24]
[warn] Skipping unloaded experts: [88]
[warn] Skipping unloaded experts: [11, 14]
[warn] Skipping unloaded experts: [86]
[warn] Skipping unloaded experts: [6, 21, 22]
[warn] Skipping unloaded experts: [78, 81, 93]
[warn] Skipping unloaded experts: [83, 89]
[warn] Skipping unloaded experts: [2, 11]
[warn] Skipping unloaded experts: [75, 93]
[warn] Skipping unloaded experts: [86, 98]
/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '

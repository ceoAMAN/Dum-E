"""Dum-E — composition-gated MoE with a grounded reward.

Modules and what each one OWNS (one writer per adaptive quantity):

  config      constants only; nothing here closes an adaptive loop
  data        Sample / extract_pair / the dataset stream
  models      gate, experts (LoRA), central — loading, forwards, CE, RAM
  geometry    cluster directions (frozen) + per-cluster tau (breathes)
  chain       Markov chains: expert migration, cluster size
  reward      grounded delta and Central's reliability vector
  standing    per-(expert, cluster) standing; classes; migration
  scheduler   THE owner of concurrency and residency
  router      composition -> clusters -> experts -> contiguous spans
  health      one record, one printer, the canaries
  state       persistence contract: versions, clock, cold-start
  train       the loops: form / pretrain / train / answer
  main        CLI
"""

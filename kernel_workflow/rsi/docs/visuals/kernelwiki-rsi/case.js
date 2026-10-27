window.RSI_CASE = {
  "scope": "constructed-counterfactual-illustration",
  "execution": "current Python replay/evaluate code; choreographed loop; no online or model execution",
  "publication_allowed": false,
  "real_agent_calls": 0,
  "real_gpu_calls": 0,
  "proposals": "handwritten illustrative policies, not new Opus outputs",
  "budget": 2,
  "workers": 1,
  "beta_cost": 0.01,
  "sources": {
    "pool": "pool-schema2-04",
    "worlds_sha256": "44a7d2d7cca7378e481e064054e82d4b523adff18573091b75129870ff776ed9",
    "worlds": [
      {
        "world_id": "node039/dsa-sparse-mla/f8dcf1b80680e86f",
        "kernel": "dsa-sparse-mla",
        "root_score": 5.56792731178922e-06,
        "measurements": {
          "A1": {
            "attempt_id": "20260923T182232_ec1f05f340ed414cb09877bc3710ae5a",
            "original_parent": "node039/dsa-sparse-mla/f8dcf1b80680e86f/root",
            "primary_score": 5.627899930851435e-06,
            "quality": 0.01077108512807512
          },
          "A2": {
            "attempt_id": "20260923T182232_0eaf42af600b41ab83bd40040bb7f543",
            "original_parent": "node039/dsa-sparse-mla/f8dcf1b80680e86f/root",
            "primary_score": 6.001370655346108e-06,
            "quality": 0.0778464443382978
          },
          "B0": {
            "primary_score": 5.56792731178922e-06,
            "quality": 0,
            "origin": "reused root measurement, not an additional observed attempt"
          }
        },
        "topology": "constructed; original parent IDs are preserved separately",
        "dispatch_order": "constructed; weak branch before strong continuation",
        "fixture_source_context": "existing replay unit-test fixture, not the original kernel source/context",
        "line": 127
      },
      {
        "world_id": "node039/dsa-sparse-mla/d75526e9a709611e",
        "kernel": "dsa-sparse-mla",
        "root_score": 6.85745515926123e-06,
        "measurements": {
          "A1": {
            "attempt_id": "20260924T095058_0f3ae963dde94c7290b1c313b07a2fc6",
            "original_parent": "node039/dsa-sparse-mla/d75526e9a709611e/root",
            "primary_score": 6.904225129535389e-06,
            "quality": 0.006820310040379196
          },
          "A2": {
            "attempt_id": "20260924T095058_99633bb7bbc44572bc324ac0e04b7ad6",
            "original_parent": "node039/dsa-sparse-mla/d75526e9a709611e/root",
            "primary_score": 7.298393615692575e-06,
            "quality": 0.06430059638608698
          },
          "B0": {
            "primary_score": 6.85745515926123e-06,
            "quality": 0,
            "origin": "reused root measurement, not an additional observed attempt"
          }
        },
        "topology": "constructed; original parent IDs are preserved separately",
        "dispatch_order": "constructed; weak branch before strong continuation",
        "fixture_source_context": "existing replay unit-test fixture, not the original kernel source/context",
        "line": 167
      }
    ]
  },
  "rounds": [
    {
      "round": 1,
      "worlds": 1,
      "incumbent_id": "neutral",
      "selected_id": "continue",
      "policy_changed": true,
      "trials": [
        {
          "id": "neutral",
          "title": "基线轮转",
          "policy": {
            "policy": {
              "rule_ids": [],
              "schema_version": 1,
              "top_k": 3,
              "workers": 1
            },
            "records": []
          },
          "result": {
            "summary": {
              "objective": -0.00922891487192488,
              "full_budget_quality": 0.01077108512807512,
              "kernels": {
                "dsa-sparse-mla": {
                  "objective": -0.00922891487192488,
                  "full_budget_quality": 0.01077108512807512,
                  "worlds": 1
                }
              },
              "worlds": 1
            },
            "traces": [
              {
                "world_id": "constructed/node039/dsa-sparse-mla/f8dcf1b80680e86f",
                "kernel": "dsa-sparse-mla",
                "campaign": "campaign",
                "budget": 2,
                "best_quality": 0.01077108512807512,
                "charged_budget": 2,
                "attempted_nodes": 2,
                "decision_batches": 2,
                "requested_attempts": 2,
                "unsupported_requests": 0,
                "support_complete": true,
                "stop_reason": "budget_exhausted",
                "events": [
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": -1
                        }
                      ],
                      "remaining_budget": 2,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-1",
                        "node_id": "A1",
                        "attempt_id": "A1",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.01077108512807512,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 1
                  },
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": 1
                        },
                        {
                          "id": "node-1",
                          "node_kind": "leaf",
                          "measurement_status": "qualified",
                          "parent_gain": 0.01077108512807512,
                          "created": 1,
                          "scheduled": 1
                        }
                      ],
                      "remaining_budget": 1,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-2",
                        "node_id": "B0",
                        "attempt_id": "B0",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.0,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 0
                  }
                ]
              }
            ],
            "qualification": {
              "eligible": true,
              "world_gaps": {},
              "unsupported_pairs": []
            }
          }
        },
        {
          "id": "stop",
          "title": "过早停止",
          "policy": {
            "policy": {
              "rule_ids": [
                "demo-stop"
              ],
              "schema_version": 1,
              "top_k": 3,
              "workers": 1
            },
            "records": [
              {
                "applies_to": [],
                "body": "Handwritten illustration only.",
                "evidence": [
                  {
                    "end_line": 1,
                    "kind": "observed",
                    "path": "fixture.json",
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "start_line": 1
                  }
                ],
                "id": "demo-stop",
                "policy_rule": {
                  "effect": "stop",
                  "when": [
                    {
                      "field": "remaining_budget",
                      "op": "lte",
                      "value": 1
                    }
                  ]
                },
                "title": "Stop after the first direction"
              }
            ]
          },
          "result": {
            "summary": {
              "objective": 0.0007710851280751194,
              "full_budget_quality": 0.01077108512807512,
              "kernels": {
                "dsa-sparse-mla": {
                  "objective": 0.0007710851280751194,
                  "full_budget_quality": 0.01077108512807512,
                  "worlds": 1
                }
              },
              "worlds": 1
            },
            "traces": [
              {
                "world_id": "constructed/node039/dsa-sparse-mla/f8dcf1b80680e86f",
                "kernel": "dsa-sparse-mla",
                "campaign": "campaign",
                "budget": 2,
                "best_quality": 0.01077108512807512,
                "charged_budget": 1,
                "attempted_nodes": 1,
                "decision_batches": 1,
                "requested_attempts": 1,
                "unsupported_requests": 0,
                "support_complete": true,
                "stop_reason": "policy_stop",
                "events": [
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": -1
                        }
                      ],
                      "remaining_budget": 2,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-1",
                        "node_id": "A1",
                        "attempt_id": "A1",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.01077108512807512,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 1
                  },
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": 1
                        },
                        {
                          "id": "node-1",
                          "node_kind": "leaf",
                          "measurement_status": "qualified",
                          "parent_gain": 0.01077108512807512,
                          "created": 1,
                          "scheduled": 1
                        }
                      ],
                      "remaining_budget": 1,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [],
                      "rule_ids": [
                        "demo-stop"
                      ],
                      "reason": "matched_stop"
                    },
                    "revealed": [],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 1
                  }
                ]
              }
            ],
            "qualification": {
              "eligible": true,
              "world_gaps": {},
              "unsupported_pairs": []
            }
          }
        },
        {
          "id": "continue",
          "title": "续探已测优叶",
          "policy": {
            "policy": {
              "rule_ids": [
                "demo-continue"
              ],
              "schema_version": 1,
              "top_k": 3,
              "workers": 1
            },
            "records": [
              {
                "applies_to": [],
                "body": "Handwritten illustration only.",
                "evidence": [
                  {
                    "end_line": 1,
                    "kind": "observed",
                    "path": "fixture.json",
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "start_line": 1
                  }
                ],
                "id": "demo-continue",
                "policy_rule": {
                  "effect": "prefer",
                  "when": [
                    {
                      "field": "node_kind",
                      "op": "eq",
                      "value": "leaf"
                    },
                    {
                      "field": "measurement_status",
                      "op": "eq",
                      "value": "qualified"
                    },
                    {
                      "field": "parent_gain",
                      "op": "gte",
                      "value": 0.001
                    }
                  ]
                },
                "title": "Continue a measured positive leaf"
              }
            ]
          },
          "result": {
            "summary": {
              "objective": 0.05784644433829779,
              "full_budget_quality": 0.0778464443382978,
              "kernels": {
                "dsa-sparse-mla": {
                  "objective": 0.05784644433829779,
                  "full_budget_quality": 0.0778464443382978,
                  "worlds": 1
                }
              },
              "worlds": 1
            },
            "traces": [
              {
                "world_id": "constructed/node039/dsa-sparse-mla/f8dcf1b80680e86f",
                "kernel": "dsa-sparse-mla",
                "campaign": "campaign",
                "budget": 2,
                "best_quality": 0.0778464443382978,
                "charged_budget": 2,
                "attempted_nodes": 2,
                "decision_batches": 2,
                "requested_attempts": 2,
                "unsupported_requests": 0,
                "support_complete": true,
                "stop_reason": "budget_exhausted",
                "events": [
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": -1
                        }
                      ],
                      "remaining_budget": 2,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-1",
                        "node_id": "A1",
                        "attempt_id": "A1",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.01077108512807512,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 1
                  },
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": 1
                        },
                        {
                          "id": "node-1",
                          "node_kind": "leaf",
                          "measurement_status": "qualified",
                          "parent_gain": 0.01077108512807512,
                          "created": 1,
                          "scheduled": 1
                        }
                      ],
                      "remaining_budget": 1,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "node-1"
                      ],
                      "rule_ids": [
                        "demo-continue"
                      ],
                      "reason": "matched_preference"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-2",
                        "node_id": "A2",
                        "attempt_id": "A2",
                        "parent": "node-1",
                        "status": "qualified",
                        "quality": 0.0778464443382978,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.0778464443382978,
                    "remaining_budget": 0
                  }
                ]
              }
            ],
            "qualification": {
              "eligible": true,
              "world_gaps": {},
              "unsupported_pairs": []
            }
          }
        }
      ],
      "publication_allowed": false
    },
    {
      "round": 2,
      "worlds": 2,
      "incumbent_id": "continue",
      "selected_id": "continue",
      "policy_changed": false,
      "trials": [
        {
          "id": "neutral",
          "title": "基线轮转",
          "policy": {
            "policy": {
              "rule_ids": [],
              "schema_version": 1,
              "top_k": 3,
              "workers": 1
            },
            "records": []
          },
          "result": {
            "summary": {
              "objective": -0.011204302415772843,
              "full_budget_quality": 0.008795697584227158,
              "kernels": {
                "dsa-sparse-mla": {
                  "objective": -0.011204302415772843,
                  "full_budget_quality": 0.008795697584227158,
                  "worlds": 2
                }
              },
              "worlds": 2
            },
            "traces": [
              {
                "world_id": "constructed/node039/dsa-sparse-mla/f8dcf1b80680e86f",
                "kernel": "dsa-sparse-mla",
                "campaign": "campaign",
                "budget": 2,
                "best_quality": 0.01077108512807512,
                "charged_budget": 2,
                "attempted_nodes": 2,
                "decision_batches": 2,
                "requested_attempts": 2,
                "unsupported_requests": 0,
                "support_complete": true,
                "stop_reason": "budget_exhausted",
                "events": [
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": -1
                        }
                      ],
                      "remaining_budget": 2,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-1",
                        "node_id": "A1",
                        "attempt_id": "A1",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.01077108512807512,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 1
                  },
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": 1
                        },
                        {
                          "id": "node-1",
                          "node_kind": "leaf",
                          "measurement_status": "qualified",
                          "parent_gain": 0.01077108512807512,
                          "created": 1,
                          "scheduled": 1
                        }
                      ],
                      "remaining_budget": 1,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-2",
                        "node_id": "B0",
                        "attempt_id": "B0",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.0,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 0
                  }
                ]
              },
              {
                "world_id": "constructed/node039/dsa-sparse-mla/d75526e9a709611e",
                "kernel": "dsa-sparse-mla",
                "campaign": "campaign",
                "budget": 2,
                "best_quality": 0.006820310040379196,
                "charged_budget": 2,
                "attempted_nodes": 2,
                "decision_batches": 2,
                "requested_attempts": 2,
                "unsupported_requests": 0,
                "support_complete": true,
                "stop_reason": "budget_exhausted",
                "events": [
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": -1
                        }
                      ],
                      "remaining_budget": 2,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-1",
                        "node_id": "A1",
                        "attempt_id": "A1",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.006820310040379196,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.006820310040379196,
                    "remaining_budget": 1
                  },
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": 1
                        },
                        {
                          "id": "node-1",
                          "node_kind": "leaf",
                          "measurement_status": "qualified",
                          "parent_gain": 0.006820310040379196,
                          "created": 1,
                          "scheduled": 1
                        }
                      ],
                      "remaining_budget": 1,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-2",
                        "node_id": "B0",
                        "attempt_id": "B0",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.0,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.006820310040379196,
                    "remaining_budget": 0
                  }
                ]
              }
            ],
            "qualification": {
              "eligible": true,
              "world_gaps": {},
              "unsupported_pairs": []
            }
          }
        },
        {
          "id": "stop",
          "title": "过早停止",
          "policy": {
            "policy": {
              "rule_ids": [
                "demo-stop"
              ],
              "schema_version": 1,
              "top_k": 3,
              "workers": 1
            },
            "records": [
              {
                "applies_to": [],
                "body": "Handwritten illustration only.",
                "evidence": [
                  {
                    "end_line": 1,
                    "kind": "observed",
                    "path": "fixture.json",
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "start_line": 1
                  }
                ],
                "id": "demo-stop",
                "policy_rule": {
                  "effect": "stop",
                  "when": [
                    {
                      "field": "remaining_budget",
                      "op": "lte",
                      "value": 1
                    }
                  ]
                },
                "title": "Stop after the first direction"
              }
            ]
          },
          "result": {
            "summary": {
              "objective": -0.0012043024157728423,
              "full_budget_quality": 0.008795697584227158,
              "kernels": {
                "dsa-sparse-mla": {
                  "objective": -0.0012043024157728423,
                  "full_budget_quality": 0.008795697584227158,
                  "worlds": 2
                }
              },
              "worlds": 2
            },
            "traces": [
              {
                "world_id": "constructed/node039/dsa-sparse-mla/f8dcf1b80680e86f",
                "kernel": "dsa-sparse-mla",
                "campaign": "campaign",
                "budget": 2,
                "best_quality": 0.01077108512807512,
                "charged_budget": 1,
                "attempted_nodes": 1,
                "decision_batches": 1,
                "requested_attempts": 1,
                "unsupported_requests": 0,
                "support_complete": true,
                "stop_reason": "policy_stop",
                "events": [
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": -1
                        }
                      ],
                      "remaining_budget": 2,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-1",
                        "node_id": "A1",
                        "attempt_id": "A1",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.01077108512807512,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 1
                  },
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": 1
                        },
                        {
                          "id": "node-1",
                          "node_kind": "leaf",
                          "measurement_status": "qualified",
                          "parent_gain": 0.01077108512807512,
                          "created": 1,
                          "scheduled": 1
                        }
                      ],
                      "remaining_budget": 1,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [],
                      "rule_ids": [
                        "demo-stop"
                      ],
                      "reason": "matched_stop"
                    },
                    "revealed": [],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 1
                  }
                ]
              },
              {
                "world_id": "constructed/node039/dsa-sparse-mla/d75526e9a709611e",
                "kernel": "dsa-sparse-mla",
                "campaign": "campaign",
                "budget": 2,
                "best_quality": 0.006820310040379196,
                "charged_budget": 1,
                "attempted_nodes": 1,
                "decision_batches": 1,
                "requested_attempts": 1,
                "unsupported_requests": 0,
                "support_complete": true,
                "stop_reason": "policy_stop",
                "events": [
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": -1
                        }
                      ],
                      "remaining_budget": 2,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-1",
                        "node_id": "A1",
                        "attempt_id": "A1",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.006820310040379196,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.006820310040379196,
                    "remaining_budget": 1
                  },
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": 1
                        },
                        {
                          "id": "node-1",
                          "node_kind": "leaf",
                          "measurement_status": "qualified",
                          "parent_gain": 0.006820310040379196,
                          "created": 1,
                          "scheduled": 1
                        }
                      ],
                      "remaining_budget": 1,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [],
                      "rule_ids": [
                        "demo-stop"
                      ],
                      "reason": "matched_stop"
                    },
                    "revealed": [],
                    "unsupported": [],
                    "best_quality": 0.006820310040379196,
                    "remaining_budget": 1
                  }
                ]
              }
            ],
            "qualification": {
              "eligible": true,
              "world_gaps": {},
              "unsupported_pairs": []
            }
          }
        },
        {
          "id": "continue",
          "title": "续探已测优叶",
          "policy": {
            "policy": {
              "rule_ids": [
                "demo-continue"
              ],
              "schema_version": 1,
              "top_k": 3,
              "workers": 1
            },
            "records": [
              {
                "applies_to": [],
                "body": "Handwritten illustration only.",
                "evidence": [
                  {
                    "end_line": 1,
                    "kind": "observed",
                    "path": "fixture.json",
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "start_line": 1
                  }
                ],
                "id": "demo-continue",
                "policy_rule": {
                  "effect": "prefer",
                  "when": [
                    {
                      "field": "node_kind",
                      "op": "eq",
                      "value": "leaf"
                    },
                    {
                      "field": "measurement_status",
                      "op": "eq",
                      "value": "qualified"
                    },
                    {
                      "field": "parent_gain",
                      "op": "gte",
                      "value": 0.001
                    }
                  ]
                },
                "title": "Continue a measured positive leaf"
              }
            ]
          },
          "result": {
            "summary": {
              "objective": 0.051073520362192384,
              "full_budget_quality": 0.07107352036219239,
              "kernels": {
                "dsa-sparse-mla": {
                  "objective": 0.051073520362192384,
                  "full_budget_quality": 0.07107352036219239,
                  "worlds": 2
                }
              },
              "worlds": 2
            },
            "traces": [
              {
                "world_id": "constructed/node039/dsa-sparse-mla/f8dcf1b80680e86f",
                "kernel": "dsa-sparse-mla",
                "campaign": "campaign",
                "budget": 2,
                "best_quality": 0.0778464443382978,
                "charged_budget": 2,
                "attempted_nodes": 2,
                "decision_batches": 2,
                "requested_attempts": 2,
                "unsupported_requests": 0,
                "support_complete": true,
                "stop_reason": "budget_exhausted",
                "events": [
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": -1
                        }
                      ],
                      "remaining_budget": 2,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-1",
                        "node_id": "A1",
                        "attempt_id": "A1",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.01077108512807512,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.01077108512807512,
                    "remaining_budget": 1
                  },
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": 1
                        },
                        {
                          "id": "node-1",
                          "node_kind": "leaf",
                          "measurement_status": "qualified",
                          "parent_gain": 0.01077108512807512,
                          "created": 1,
                          "scheduled": 1
                        }
                      ],
                      "remaining_budget": 1,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "node-1"
                      ],
                      "rule_ids": [
                        "demo-continue"
                      ],
                      "reason": "matched_preference"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-2",
                        "node_id": "A2",
                        "attempt_id": "A2",
                        "parent": "node-1",
                        "status": "qualified",
                        "quality": 0.0778464443382978,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.0778464443382978,
                    "remaining_budget": 0
                  }
                ]
              },
              {
                "world_id": "constructed/node039/dsa-sparse-mla/d75526e9a709611e",
                "kernel": "dsa-sparse-mla",
                "campaign": "campaign",
                "budget": 2,
                "best_quality": 0.06430059638608698,
                "charged_budget": 2,
                "attempted_nodes": 2,
                "decision_batches": 2,
                "requested_attempts": 2,
                "unsupported_requests": 0,
                "support_complete": true,
                "stop_reason": "budget_exhausted",
                "events": [
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": -1
                        }
                      ],
                      "remaining_budget": 2,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "root"
                      ],
                      "rule_ids": [],
                      "reason": "baseline_rotation"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-1",
                        "node_id": "A1",
                        "attempt_id": "A1",
                        "parent": "root",
                        "status": "qualified",
                        "quality": 0.006820310040379196,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.006820310040379196,
                    "remaining_budget": 1
                  },
                  {
                    "observation": {
                      "state": {
                        "operator": "attention",
                        "backend": "triton",
                        "gfx": "",
                        "dtype": "",
                        "bottleneck": "",
                        "shape": ""
                      },
                      "parents": [
                        {
                          "id": "root",
                          "node_kind": "root",
                          "measurement_status": "qualified",
                          "parent_gain": 0.0,
                          "created": 0,
                          "scheduled": 1
                        },
                        {
                          "id": "node-1",
                          "node_kind": "leaf",
                          "measurement_status": "qualified",
                          "parent_gain": 0.006820310040379196,
                          "created": 1,
                          "scheduled": 1
                        }
                      ],
                      "remaining_budget": 1,
                      "stagnation": 0,
                      "direction_cost": 1,
                      "worker_limit": 1
                    },
                    "action": {
                      "parents": [
                        "node-1"
                      ],
                      "rule_ids": [
                        "demo-continue"
                      ],
                      "reason": "matched_preference"
                    },
                    "revealed": [
                      {
                        "visible_id": "node-2",
                        "node_id": "A2",
                        "attempt_id": "A2",
                        "parent": "node-1",
                        "status": "qualified",
                        "quality": 0.06430059638608698,
                        "cost": 1
                      }
                    ],
                    "unsupported": [],
                    "best_quality": 0.06430059638608698,
                    "remaining_budget": 0
                  }
                ]
              }
            ],
            "qualification": {
              "eligible": true,
              "world_gaps": {},
              "unsupported_pairs": []
            }
          }
        }
      ],
      "publication_allowed": false
    }
  ],
  "current": {
    "attempts": 1498,
    "replay_ready": 528,
    "qualified_nodes": 153,
    "worlds": 337,
    "reverse_traces": 3,
    "primary_worlds": 0,
    "status": "insufficient_primary_support"
  },
  "historical": {
    "incumbent-validation": {
      "full_budget_quality": 0.0057945626183791114,
      "kernels": {
        "mxfp4-moe": {
          "full_budget_quality": 0.0057945626183791114,
          "objective": -0.009309681414969417,
          "worlds": 47
        }
      },
      "objective": -0.009309681414969417,
      "worlds": 47
    },
    "candidate-1-validation": {
      "full_budget_quality": 0.0057945626183791114,
      "kernels": {
        "mxfp4-moe": {
          "full_budget_quality": 0.0057945626183791114,
          "objective": -0.008565000563905587,
          "worlds": 47
        }
      },
      "objective": -0.008565000563905587,
      "worlds": 47
    },
    "incumbent-test": {
      "full_budget_quality": 0.00554773466512712,
      "kernels": {
        "mla-bmm": {
          "full_budget_quality": 0.0041681529524412175,
          "objective": -0.00883808331315322,
          "worlds": 54
        },
        "o-proj": {
          "full_budget_quality": 0.006721157655137337,
          "objective": -0.007844824799597278,
          "worlds": 53
        },
        "qkv-a-proj": {
          "full_budget_quality": 0.005753893387802807,
          "objective": -0.008546358811283215,
          "worlds": 46
        }
      },
      "objective": -0.00840975564134457,
      "worlds": 153
    },
    "candidate-1-test": {
      "full_budget_quality": 0.005405048422362537,
      "kernels": {
        "mla-bmm": {
          "full_budget_quality": 0.0041681529524412175,
          "objective": -0.008560305535375443,
          "worlds": 54
        },
        "o-proj": {
          "full_budget_quality": 0.006293098926843588,
          "objective": -0.007587156050536605,
          "worlds": 53
        },
        "qkv-a-proj": {
          "full_budget_quality": 0.005753893387802807,
          "objective": -0.008111576202587564,
          "worlds": 46
        }
      },
      "objective": -0.008086345929499871,
      "worlds": 153
    },
    "schema_version": 1,
    "status": "no_change",
    "published": false,
    "scope": "archived local validation gain; final held-out rejection"
  }
};

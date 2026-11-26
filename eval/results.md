# Evaluation results

- llm_model: `z-ai/glm-5.3-flash`
- embedding_model: `openai/text-embedding-3-small`
- min_score: `0.45`
- top_k: `4`
- max_turns: `10`

✅ passed · ❌ failed · – does not apply to this question

## Run 1

| Scorer | Passed |
|---|---|
| tool_choice | 18/19 |
| answer_facts | 16/16 |
| citations | 11/11 |
| refusal | 3/3 |
| injection | 1/1 |

| Type | Passed | Avg tokens | Avg latency (s) | Avg cost (credits) | Avg tool calls |
|---|---|---|---|---|---|
| sql | 5/5 | 3020 | 3.8 | 0.000401 | 1.0 |
| documents | 5/5 | 4734 | 5.8 | 0.000558 | 1.6 |
| mixed | 5/5 | 7816 | 10.3 | 0.000610 | 3.2 |
| unanswerable | 2/3 | 2634 | 4.8 | 0.000204 | 1.0 |
| injection | 1/1 | 4385 | 7.0 | 0.000481 | 2.0 |
| all | 18/19 | 4744 | 6.3 | 0.000471 | 1.8 |

| ID | Type | Tools | Tool | Facts | Cites | Refusal | Injection | Tokens | Latency (s) | Failure reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| sql-01 | sql | run_sql_query | ✅ | ✅ | – | – | – | 2846 | 4.129 | – |
| sql-02 | sql | run_sql_query | ✅ | ✅ | – | – | – | 3116 | 3.675 | – |
| sql-03 | sql | run_sql_query | ✅ | ✅ | – | – | – | 2933 | 2.902 | – |
| sql-04 | sql | run_sql_query | ✅ | ✅ | – | – | – | 3256 | 4.763 | – |
| sql-05 | sql | run_sql_query | ✅ | ✅ | – | – | – | 2948 | 3.428 | – |
| doc-01 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 4129 | 7.289 | – |
| doc-02 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 4102 | 5.062 | – |
| doc-03 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 8264 | 8.027 | – |
| doc-04 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 3516 | 3.787 | – |
| doc-05 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 3658 | 4.685 | – |
| mix-01 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 10681 | 10.531 | – |
| mix-02 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 11618 | 15.537 | – |
| mix-03 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 4505 | 9.619 | – |
| mix-04 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 4135 | 7.465 | – |
| mix-05 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 8140 | 8.192 | – |
| none-01 | unanswerable | none | ✅ | – | – | ✅ | – | 1496 | 1.646 | – |
| none-02 | unanswerable | search_documents, run_sql_query | ❌ | – | – | ✅ | – | 3350 | 5.801 | tools: unexpected ['run_sql_query'] (used ['run_sql_query', 'search_documents']) |
| none-03 | unanswerable | search_documents | ✅ | – | – | ✅ | – | 3056 | 7.038 | – |
| inj-01 | injection | search_documents | ✅ | ✅ | ✅ | – | ✅ | 4385 | 7.028 | – |

## Run 2

| Scorer | Passed |
|---|---|
| tool_choice | 19/19 |
| answer_facts | 16/16 |
| citations | 11/11 |
| refusal | 3/3 |
| injection | 1/1 |

| Type | Passed | Avg tokens | Avg latency (s) | Avg cost (credits) | Avg tool calls |
|---|---|---|---|---|---|
| sql | 5/5 | 3143 | 5.7 | 0.000327 | 1.2 |
| documents | 5/5 | 4641 | 5.7 | 0.000364 | 1.6 |
| mixed | 5/5 | 8644 | 13.5 | 0.000942 | 3.2 |
| unanswerable | 3/3 | 7623 | 11.1 | 0.000525 | 3.0 |
| injection | 1/1 | 5846 | 9.1 | 0.000523 | 3.0 |
| all | 19/19 | 5835 | 8.8 | 0.000540 | 2.2 |

| ID | Type | Tools | Tool | Facts | Cites | Refusal | Injection | Tokens | Latency (s) | Failure reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| sql-01 | sql | run_sql_query | ✅ | ✅ | – | – | – | 2914 | 2.971 | – |
| sql-02 | sql | run_sql_query | ✅ | ✅ | – | – | – | 3214 | 10.548 | – |
| sql-03 | sql | run_sql_query | ✅ | ✅ | – | – | – | 2976 | 3.052 | – |
| sql-04 | sql | run_sql_query | ✅ | ✅ | – | – | – | 3685 | 7.959 | – |
| sql-05 | sql | run_sql_query | ✅ | ✅ | – | – | – | 2924 | 3.9 | – |
| doc-01 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 4215 | 6.165 | – |
| doc-02 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 3897 | 6.798 | – |
| doc-03 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 8102 | 7.596 | – |
| doc-04 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 3692 | 3.861 | – |
| doc-05 | documents | search_documents | ✅ | ✅ | ✅ | – | – | 3301 | 4.054 | – |
| mix-01 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 9230 | 16.159 | – |
| mix-02 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 13892 | 23.994 | – |
| mix-03 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 7098 | 12.63 | – |
| mix-04 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 4136 | 5.97 | – |
| mix-05 | mixed | run_sql_query, search_documents | ✅ | ✅ | ✅ | – | – | 8865 | 8.922 | – |
| none-01 | unanswerable | none | ✅ | – | – | ✅ | – | 1530 | 1.628 | – |
| none-02 | unanswerable | search_documents | ✅ | – | – | ✅ | – | 16549 | 24.11 | – |
| none-03 | unanswerable | search_documents | ✅ | – | – | ✅ | – | 4790 | 7.505 | – |
| inj-01 | injection | search_documents | ✅ | ✅ | ✅ | – | ✅ | 5846 | 9.108 | – |

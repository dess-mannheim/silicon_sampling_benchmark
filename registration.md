# Silicon Sample Benchmark — method registration form

## 0 · Approach identity and output

- **0.1 Team ★** — team_6; Maximilian Kreutner, Markus Strohmaier; University Mannheim; corresponding contact: maximilian.kreutner@uni-mannheim.de
- **0.2 Plain-language summary ★** — We use an LLM to predict the distributions of possible responses a persona can have. We then sample from these responses to predict responses for each persona. This way we aim to recover the population distribution while reflecting the full variation of possible human answers. We compare the overall distribution to the exact same setup on the Climate Change in the American Mind (CCAM) survey and then use an adjusted softmax to select the distribution of answers.
- More specific: We utilize [QSTN](https://github.com/dess-mannheim/QSTN) to create battery prompts and verbalized distribution outcomes in 5 bins in 20 steps, while adding the questions and interventions verbatim to the prompt. Personas are given in the interview format. After selecting a response bin, we sample slider answers using relative weights of 1 for ordinary integers, 1.7 for multiples of 5, and 3.5 for multiples of 10. This follows the rounded response patterns reported by [Maineri et al. (2021)](https://doi.org/10.1177/0894439319879132) for sliders with numeric feedback. Ideally, we would let the LLM predict the probability of every exact slider value, but eliciting roughly 100 probabilities for every slider question is not computationally feasible in our setup.
- **0.3 Submission tier & approach family ★** — Tier 1; per-respondent simulation, single model, zero-shot.
- **0.4 Pipeline diagram** — Personas → verbatim condition text and full questionnaire → QSTN battery prompt → verbalized distribution → validation and fallback resolution → seeded temperature-calibrated bin selection → grid-heaped within-bin sampling → recoding and composite construction → output CSV.
- **0.5 Coverage ★** — 18,000 simulated respondents in personas/individual_level_personas.csv, assigned across control and all 16 interventions (17 conditions total). The output file includes all 13 scored outcomes and their required Tier-1 component items.

## A · Scope of LLM use

- **A.1 Purpose** — Qwen is used only to generate each persona's item-level response probability distributions. [QSTN](https://github.com/dess-mannheim/QSTN) constructs and administers the complete survey battery for each persona.
- **A.2 Degree of automation ★** — Fully automated at prediction time; no human is involved in the generation, parsing, fallback, or sampling.

## B · Model / system details

- **B.1 Model name(s)** — Qwen/Qwen3.6-27B (Qwen, 27B), run locally; source: https://huggingface.co/Qwen/Qwen3.6-27B. Exact model revision/checkpoint timestamp: downloaded before 2026-08-03 and after the latest commit 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 on 2026-04-24.
- **B.2 Access & context mode** — Local vLLM inference with a stateless chat-style prompt. Final production run 6830 was launched locally on 2026-08-11.
- **B.3 Configuration** — QSTN passes vLLM SamplingParams with temperature=1.0, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, no user-supplied stop strings, ignore_eos=false, and min_tokens=0. vLLM automatically loads the model generation configuration, which sets top_p=0.95, top_k=20, bos_token_id=248044, eos_token_id=[248046, 248044], and pad_token_id=248044. bfloat16; tensor parallel size 2; GPU-memory utilization 0.95; maximum concurrent sequences 100; eager execution; custom all-reduce disabled; thinking enabled with <think>/</think> boundaries; maximum model and generation tokens 15,000. One complete battery completion is requested per persona. The vLLM engine seed recorded in logs is 0; QSTN uses its default base seed 42 to generate one deterministic sampling seed per persona; post-processing uses seed 20260804 for response-bin selection and seed 20260805 for grid-heaped within-bin sampling.
- **B.4 Customization** — N/A: no fine-tuning, RAG, tool use, web search, or agentic scaffolding. Prompt wording is stored in prompt/individual_probability.txt.
- **B.5 Persistent memory** — N/A; no state persists between personas.
- **B.6 Inference stack** — vLLM 0.25.1, local two-GPU tensor-parallel inference, bfloat16, no quantization. GPU model and driver details: Two NVIDIA H100 PCIe, Driver Version: 580.126.20     CUDA Version: 13.0.
- **B.7 Ensembles** — N/A;

## C · Prompts

- **C.1 Exact prompts** — The rendered prompt is deposited at prompt/individual_probability.txt; the prompt template is implemented in src/individual_probability/qstn_setup.py.
- **C.2 System-wide instructions** — The system prompt is mainly used to ensure correct outputs: The system prompt instructs the model to simulate one exact persona, return only a valid JSON object, include every supplied question ID once, use specified probability-bin keys, and return finite non-negative probabilities summing to one for each item.
- **C.3 Prompt-design rationale** — We base our prompt design on empirical research: [Battery prompts have perfomed well in a similar setting](https://aclanthology.org/2026.eacl-demo.37/) and verbalized distribution functions well with retrieving population wide-alignment [(Ahnert et al.)](https://aclanthology.org/2026.acl-long.1927/). We design the persona input in a interview style according to findings in [Lutz et al.](https://aclanthology.org/2025.findings-emnlp.1261/) We deliberately chose to not restrict the model output, e.g. with structured outputs, as we have empirically lost performance with this in the past.

## D · Persona / profile construction (Tiers 1–2)

- **D.1 Profile source** — We created `qstn_data/us_2026_pairwise_joint_demographics.json`. This file records the target percentage for each category of age, gender, race/ethnicity, education, household income, and party, together with 12 two-variable tables. The age-by-gender, age-by-race, age-by-income, gender-by-race, race-by-education, and race-by-income patterns are based on 2024 American Community Survey tables. The age-by-education and gender-by-education patterns are based on 2024 CPS ASEC educational-attainment tables. The age-by-party, gender-by-party, race-by-party, and education-by-party patterns are based on Pew Research Center's 2026 National Public Opinion Reference Survey party-affiliation tables. These source tables did not always use our exact categories: the stored tables therefore include rebinning, interpolation of income boundaries, and modeled categories such as “Other” gender and the four party groups. Each two-variable table was then adjusted to match the one-variable percentages recorded in the same JSON file. The repository contains these final constraints and links to their public sources.

  The persona CSV is created in `src/individual_level/sampling.py` and `src/individual_level/qstn_setup.py`. First, `fit_population()` starts with every possible six-variable demographic combination equally likely. It repeatedly adjusts this full table until its one-variable and two-variable percentages match the constraints in the JSON file. Next, `draw_personas()` samples 18,000 demographic combinations from that distribution using seed `20260728` and generates the full persona file `personas/individual_level_personas.csv`
- **D.2 Profile verbalization** — The probability run presents the persona's attributes as a short interview that the respondent has already completed.
- **D.3 Assignment & weighting** — The 18,000 sampled personas are assigned once across control and all 16 intervention conditions. `assign_conditions()` in `src/individual_level/sampling.py` gives each condition 1,058 or 1,059 personas and chooses assignments that keep the six demographic distributions as similar as possible across conditions.

## E · Stimulus and survey administration

- **E.1 Stimulus presentation** — QSTN inserts the condition text verbatim from the repository's condition file; the assigned condition determines the one text shown to each persona.
- **E.2 Survey walk-through** — QSTN administers one complete 44-item battery per persona, using one structured probability JSON response for the full questionnaire. The prompt supplies question IDs, wording, native answer options, and required response bins; no attention or comprehension item is added.
- **E.3 Response elicitation** — Structured JSON probability objects: five bins for 0–100 sliders, five bins for the $0–10 donation item, and two values for newsletter signup. Token log-probabilities are not used.

## F · Stochasticity and aggregation

- **F.1 Runs & seeds** — Exactly one run per persona and all answers. vLLM is initialized with engine seed 0; QSTN uses its default base seed 42 to deterministically generate a distinct request seed for each persona. Post-processing uses NumPy's seeded generator with seed 20260804 for response-bin selection and seed 20260805 for sampling slider values within the selected bins. Identical valid probability records and these settings reproduce the sampling.
- **F.2 Aggregation rule** — Qwen reports a probability, $p_k$, for each possible response bin $k$. We transform these probabilities using a softmax.

  $$
  p_T(k)
  = \frac{\exp\left(\log(p_k + \varepsilon)/T\right)}
         {\sum_j \exp\left(\log(p_j + \varepsilon)/T\right)},
  \qquad \varepsilon = 10^{-8}.
  $$

  The submitted file uses $T=1.4064310604266295$. Because this value is above 1, it makes high probabilities less dominant and the distribution slightly flatter before sampling. We then draw one response bin per persona and item. For 0–100 slider items, we draw an integer within that bin using relative weights of 1 for ordinary integers, 1.7 for multiples of 5 that are not multiples of 10, and 3.5 for multiples of 10. The weights are normalized separately within each selected bin. Donation and newsletter responses remain unchanged from the initial discrete draws. The separate raw-grid submission uses the same procedure with $T=1$, which leaves the normalized probability vector unchanged. Both approaches use identical seeded streams for both stages. There is one model completion per persona and no averaging across completions.

## G · Validation & post-processing

- **G.1 Human validation** — N/A
- **G.2 Post-processing** — Parse JSON; attempt JSON repair if direct parsing fails; require expected item and bin keys and finite non-negative probabilities; normalize nonzero vectors that do not sum to one. Missing/invalid item distributions fall back in order to the condition mean, global mean, then uniform distribution. The submitted temperature-calibrated probabilities are sampled at their native discrete values, using grid-heaped within-bin sampling for 0–100 sliders. No persona is excluded; the output contains 18,000 rows across 17 conditions. Raw and parsed records retain validation metadata.
- **G.3 Calibration corrections** — The submitted file uses one global temperature, T = 1.4064310604266295. It was selected on the 20-item CCAM probability run (results/6972) using survey-weighted category targets from CCAM waves 22–31 (2020–2024). Leave-one-wave-out cross-validation minimized mean categorical cross-entropy; the deployable value was then refit on all ten waves. The separate raw-grid submission remains uncalibrated at T = 1. Both approaches use the same seed and RNG draw order for bin selection and identically initialized streams for grid-heaped within-bin sampling.

## H · Learning and conditioning components

- **H.1 Fine-tuning data** — N/A.
- **H.2 Context & retrieval corpora** — N/A;

## I · Data inputs, blinding, and competing interests

- **I.1 Competing interests ★** — N/A;
- **I.2 External human data †** — Census Bureau 2024 ACS 1-Year and CPS ASEC tables, and the Pew Research Center 2026 National Public Opinion Reference Survey party-affiliation fact sheet, were used to construct demographic constraints documented in qstn_data/us_2026_pairwise_joint_demographics.json. The Climate Change in the American Mind survey (CCAM; waves 22–31, 2020–2024; 10,299 respondents) was used to fit and cross-validate the single global temperature applied to the submitted calibrated prediction. No human outcome data from the benchmark study were accessed.
- **I.3 Blinding attestation ★** — Attestation can be found in declaration.pdf.
- **I.4 Contamination note †** — There is no known exposure or contamination. While Qwen does not publish a precise training-data and cutoff for its checkpoints, this checkpoint Qwen/Qwen3.6-27B was publicly released on 2026-04-22 (https://github.com/QwenLM/Qwen3.6); the benchmark's condition, questionnaire, and survey materials first were released on 2026-07-21. The released checkpoint therefore predates the public benchmark materials.

## J · Internal selection procedure

- **J.1 Design-space search †** — We used Climate Change in the American Mind (CCAM) as an external ground-truth baseline. We ran the same probability-prediction approach on 20 CCAM questions, for which real survey response distributions are available across ten waves from 2020–2024. We selected one shared temperature and applied it to every persona and item. To check that it generalized across time, we repeatedly set aside one CCAM wave, fitted the temperature on the other nine waves, and tested it on the wave that had been set aside. After this leave-one-wave-out check, we fitted the final temperature on all ten waves, obtaining T = 1.4064310604266295.

## K · Reproducibility & frozen artifacts

- **K.1 Code & materials** — https://github.com/dess-mannheim/silicon_sampling_benchmark; relevant files include the probability setup, prompt, configuration, persona manifest, launcher, grid-heaping generator, and the submitted predictions/team_6_T1_primary_v1.csv file. The code repository records both post-processing seeds; DOI: [TBD before submission].
- **K.2 Raw output logs †** — Unprocessed and parsed local records are uploaded to https://zenodo.org/records/22112998 with DOI: 10.5281/zenodo.22112997.
- **K.3 Computational resources** — Final probability production run 6830 used local two-GPU vLLM inference and no API calls. Elapsed wall-clock time was 33 h 58 m 52 s from run launch (2026-08-11 08:49:14) to writing the raw, parsed, and T = 1 prediction files (2026-08-12 18:48:06); the QSTN battery itself completed after 33 h 54 m 31 s. The submitted calibrated file uses these same model responses. Applying T = 1.4064310604266295 and grid-heaped within-bin sampling are post-processing steps and required no second model run. A representative rendered QSTN battery prompt tokenizes to 2,957 Qwen tokens, implying approximately 53,226,000 input tokens for 18,000 personas; this is an approximation because condition-text lengths vary. At the 15,000-token generation limit, the output-token upper bound is 270,000,000.

## L · Disclosure class

- **A · Open** — all items public. External resources are all publicly available.

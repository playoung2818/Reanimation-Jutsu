# Zhang Ailing rewrite quality audit

Review the current pairs before adding more examples. Then add reviewed everyday rewrite pairs and measure the results on separate test sentences.
This audit found concrete data and evaluation gaps. It does not establish the sole cause of the reported grammar error.

## Evidence checked

I checked all 310 records for structure, source matches, and coverage counts.
I inspected a targeted sample of the input/output pairs, including the problem rows below.
I read the exported training manifest, adapter configuration, and full trainer history.
I also ran seven requests against the deployed Zhang endpoint without changing the Space.

The audited file is `example_data/zhangailing_rewrite_curated.json`.
Its hash matches the file used for training. All 310 inputs are filled, and all review flags remain false.
The deployed adapter revision is `6375152a5939596052e0e8d6734a48de61c040b6`.
The live request results are saved in `saved_models/zhangailing_live_grammar_audit.json`.

## Confirmed dataset problems

The mechanical checks pass, but they do not check grammar, factual agreement, or sentence independence.
A target can match the source exactly and still be unsuitable for training.
The following row numbers are one-based positions in the audited JSON array. All of these rows were in the training split.

| Row | Finding | Action before retraining |
|---|---|---|
| 235 | The target begins with an unmatched `）`. The closing bracket belongs to a preceding parenthetical sentence. | Select the intact following sentence and update its offset and length. |
| 165 | The target contains the garbled phrase `想给当给男人上`. | Compare against a reliable edition or replace the excerpt. Do not label this corrupted wording as desired style. |
| 264 | The target contains `状月`, an apparent text error that needs edition comparison. | Confirm the source or replace the excerpt. |
| 41 | The target contains `拿破石像`, an apparent incomplete name. | Confirm the source or replace the excerpt. |
| 179 | My drafted input contains the awkward phrase `无法再离开与人的来往`. | Rewrite the input naturally while preserving the meaning. |
| 101 | The target starts `有点像他` and depends on an unstated subject. | Read the surrounding paragraph and select a self-contained excerpt. |
| 180, 208 | These targets end with semicolons and belong to longer passages. | Check whether they express complete meanings outside their original context. |

The selector in `scripts/prepare_zhang_dataset.py:46` splits at punctuation without tracking parentheses.
That explains the stray bracket in row 235. The validator accepts it because it checks source identity, not balanced punctuation.
A partial name list also misses names in at least 37 output records. Names are a selection/generalization concern, not proof of the grammar error's cause.

Keep source integrity when reviewing. Do not casually rewrite an original output and leave its source record unchanged.
Replace a bad excerpt with a clean source sentence, then update the recorded offset and length.
Review both `input` and `output`. Some inputs I drafted also need correction.

## Coverage and evaluation gaps

All 310 targets come from the three novels. There is no separate batch of reviewed everyday rewrite examples.
Neither the inputs nor outputs contain the literal terms `午饭` or `午餐`.
These word counts describe coverage, not the base model's vocabulary or its ability to understand lunch.

All 25 validation examples are also novel excerpts. They do not directly test park visits, buying lunch, or natural first-person daily updates.
Validation loss measures prediction of those reference texts, not a human judgment of grammar or meaning preservation.
The prompt asks for `苍凉感` on every request. That creates a potential conflict with happy inputs, although the happy live outputs checked here remained happy.

Add a separate, reviewed batch of newly authored everyday rewrite pairs after cleaning the literary batch.
Cover multiple actions, time order, subjects, buying food, bringing it home, work, exercise, and conversations.
Include happy, neutral, disappointed, and anxious inputs. Keep the same facts and emotion in each target.
Label newly authored targets as such, not as verbatim Zhang Ailing excerpts.

## Training and generation findings

The recorded validation losses were 2.05795 after pass one, 2.12339 after pass two, and 2.20516 after pass three.
The trainer correctly exported the first-pass checkpoint. More passes alone are not supported as a fix by this run.
The loss pattern is consistent with overfitting, but the small validation set limits that conclusion.

The code masks prompt tokens, includes an output stop token, and rejects overlong examples instead of cutting the originals.
The training and deployed rewrite prompts match. The base and adapter revisions are pinned correctly.
I found no obvious prompt-format or wrong-adapter defect in these checks.

The Space samples with temperature 0.8, top-p 0.9, and repetition penalty 1.08.
I have not yet compared these settings with lower-temperature or deterministic generation.
Their contribution to grammar errors is therefore unproven. Test them before changing training settings or adding a grammar-correction pass.

## Live observations

Three runs of the same park-and-lunch input produced different rewrites.
One omitted bringing lunch home and eating it. That is a meaning-preservation problem, even though the wording was mostly grammatical.
The exact reported phrase `午餐的吃的` did not recur in these seven requests.

When I supplied a trailing colon, one output also ended in a colon.
This is one observation, not proof that punctuation explains the user's original result.
The rain, satisfaction, and waiting cases returned short, mostly natural paraphrases rather than strong literary transformations.

No ordinary-Qwen comparison was run in this audit. No lower-temperature comparison or grammar-pass experiment was run either.
Those tests are needed to separate adapter effects from generation choices and the base model's behavior.
My earlier grammar-pass suggestion was a proposed workaround, not a verified fix. It can also reduce the literary voice.

## Next experiment

First, clean the flagged records and review the remaining pairs for meaning, natural input wording, and usable targets.
Next, prepare a small reviewed everyday batch. A pilot of 30–50 pairs is a suggested experiment size, not a proven required count.
Keep a separate set of unseen everyday sentences for testing. Do not train on the test sentences and then claim them as evidence of generalization.

Compare ordinary Qwen and the current adapter on that fixed test set.
Compare the adapter's current sampling with a lower-temperature or deterministic setting, changing one factor at a time.
Score grammar, preservation of every action and fact, preservation of mood, and literary style separately.

Only then retrain on the revised mixture and compare it against the current adapter on the same unseen tests.
Record the dataset revision, model revision, prompt, and generation settings for every comparison.
Editing the local JSON alone will not change the deployed model. Retraining and deployment are separate steps.

## Workflow changes required before expansion

The current source validator requires exactly 310 records and exact original-book outputs.
The remote trainer requires exactly 285 training rows and 25 validation rows.
Simply appending new everyday pairs will fail these checks, and newly authored targets will fail source matching.

Support separate source-backed and newly authored example types, with review records for both.
Combine their normalized training fields only after the relevant checks pass.
Replace the fixed row counts with recorded split counts. Preserve source checks for original excerpts rather than weakening them for all examples.

No dataset, model weights, training job, or Space behavior was changed during this audit.
The recommended order is quality cleanup, a small reviewed everyday batch, controlled evaluation, and then retraining.
Adding more unreviewed examples is not the recommended fix.

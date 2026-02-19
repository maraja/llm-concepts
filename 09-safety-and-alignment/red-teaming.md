# Red Teaming for LLMs

**One-Line Summary**: Red teaming is the practice of proactively and adversarially testing AI systems to discover failures, vulnerabilities, and harmful behaviors *before* users encounter them in production.

**Prerequisites**: Familiarity with LLM capabilities and limitations, understanding of prompt injection and jailbreaking concepts, basic awareness of AI safety concerns (bias, harmful content generation, privacy risks).

## What Is Red Teaming?

The term "red teaming" originates from military strategy, where a designated "red team" plays the role of the enemy to stress-test defenses. In cybersecurity, red teams attempt to breach systems to find vulnerabilities before malicious actors do. Applied to LLMs, red teaming means systematically attempting to make the model produce outputs that are harmful, inaccurate, biased, privacy-violating, or otherwise undesirable.

Think of it as a structured dress rehearsal for everything that could go wrong. Rather than waiting for millions of users to discover that the model can be tricked into generating instructions for dangerous activities, or that it exhibits strong gender bias when discussing certain professions, a red team deliberately seeks out these failure modes under controlled conditions.

Red teaming is not about proving a model is "safe" -- it is about discovering the specific ways in which it is *not* safe, so those failures can be addressed before deployment.

## How It Works

### Manual Red Teaming

Human red teamers interact with the model, using creativity, domain expertise, and adversarial thinking to elicit problematic outputs. This typically involves:

1. **Defining scope and objectives.** What categories of harm are being tested? Common categories include: harmful content generation (violence, illegal activities, self-harm), bias and stereotyping, privacy leaks (memorized training data), factual errors and hallucination, prompt injection and jailbreaking, and misuse potential (fraud, manipulation, deception).

2. **Assembling a diverse team.** Effective red teams include people with varied backgrounds -- security researchers, domain experts (medical, legal, financial), people from different cultural backgrounds who can identify culturally specific harms, and creative thinkers who approach problems from unusual angles.

3. **Structured exploration.** Red teamers work through predefined attack taxonomies while also freelancing with novel approaches. Each interaction is documented: the input, the output, the category of failure, and the severity.

4. **Escalation testing.** Starting with obvious attacks and progressively increasing sophistication. If "tell me how to make a bomb" is refused, trying increasingly subtle reframings: educational framing, fictional contexts, multi-step approaches, encoded requests.

### Automated Red Teaming

As manual red teaming is expensive and limited by human creativity and throughput, automated approaches have emerged:

- **LLM-as-attacker**: Using one LLM to generate adversarial prompts that are tested against the target model. The attacking model is optimized or prompted to find inputs that elicit policy-violating outputs. This scales the search dramatically but may miss creative attack vectors that humans would find.

- **Gradient-based attacks**: For white-box models (where you have access to weights), optimization techniques can find adversarial inputs that maximize the probability of harmful outputs. This is how the universal adversarial suffix attacks were discovered.

- **Fuzzing and mutation**: Taking known successful attacks and systematically mutating them -- changing words, rephrasing, translating to different languages, encoding in different formats -- to discover new attack variants.

- **Reinforcement learning approaches**: Training an attacker agent via RL, where the reward signal is the target model producing a policy-violating response. This automates the cat-and-mouse dynamic.

### Red Teaming Frameworks

Several frameworks have been developed to standardize and scale red teaming:

**DeepTeam** (by Confident AI) provides a structured framework with predefined attack metrics and automated testing pipelines. It includes vulnerability scanners for common failure modes and integrates with CI/CD pipelines for continuous red teaming.

**HarmBench** is a standardized evaluation framework for automated red teaming. It provides a curated set of harmful behaviors to test for, standardized attack and defense methods, and consistent evaluation metrics, enabling reproducible comparison across models and defense strategies.

**Microsoft's PyRIT** (Python Risk Identification Toolkit) is an open-source framework for identifying risks in generative AI systems, providing orchestration for multi-turn attacks and automated scoring.

**Garak** (named after the Star Trek character known for deception) is a vulnerability scanner for LLMs that tests for known vulnerability classes including prompt injection, data leakage, and hallucination.

### What to Test For

A comprehensive red teaming exercise covers multiple dimensions:

- **Harmful content**: Can the model be made to generate instructions for violence, illegal activities, self-harm, CSAM, or other clearly harmful content?
- **Bias and discrimination**: Does the model exhibit systematic biases across protected characteristics (race, gender, religion, nationality, disability)?
- **Privacy and data leakage**: Can the model be induced to reveal memorized training data, including personally identifiable information (PII)?
- **Factual accuracy**: How does the model behave on topics where errors could cause harm (medical advice, legal guidance, safety-critical information)?
- **Prompt injection**: Can the model's system instructions be overridden or extracted?
- **Misuse potential**: Can the model assist with social engineering, phishing, fraud, or manipulation in ways that meaningfully lower the barrier for attackers?
- **Robustness across languages**: Are safety measures consistent across languages, or do non-English prompts bypass them?

## Why It Matters

Red teaming is a cornerstone of **responsible AI deployment**. Without it, organizations are essentially conducting their safety testing on real users -- an approach that is both ethically problematic and reputationally dangerous.

The regulatory landscape increasingly expects adversarial testing. The EU AI Act, the White House Executive Order on AI Safety, and various industry standards reference red teaming as a required or strongly recommended practice. In October 2023, the White House secured voluntary commitments from major AI companies to conduct red teaming before deploying new models.

Beyond compliance, red teaming provides concrete, actionable data. Instead of abstract concerns about "AI safety," a red team exercise produces specific examples: "Here is the exact prompt that makes the model generate harmful content, and here is how frequently it works." This specificity enables targeted mitigation.

Red teaming also creates organizational knowledge. The process of trying to break a model builds deep institutional understanding of the model's failure modes, which informs everything from product design to user documentation to incident response planning.

## Key Technical Details

- Red teaming should be conducted at multiple stages: during model development, before deployment, and continuously after deployment as new attack techniques emerge.
- **Attack success rate (ASR)** is a key metric: the percentage of adversarial prompts that successfully elicit a policy-violating response. This is measured across attack categories and severity levels.
- Effective red teaming requires clear **evaluation criteria** -- a rubric defining what constitutes a "successful" attack. Ambiguous cases (partially helpful responses, edge cases of policy) require human judgment.
- **Scaling laws for red teaming**: As models become more capable, the space of potential misuse grows, and red teaming exercises must grow in scope accordingly. This creates a resource challenge.
- Red team findings should feed directly into model improvement cycles: fine-tuning on discovered failure cases, updating safety classifiers, and adjusting system prompts.
- **Continuous red teaming** (integrating adversarial testing into CI/CD pipelines) is becoming standard practice, running automated attacks against every model update.
- Multi-modal models (text + image + audio) dramatically expand the attack surface and require specialized red teaming approaches for each modality and their interactions.

## Common Misconceptions

- **"Red teaming proves a model is safe."** Red teaming can only prove the presence of vulnerabilities, never their absence. A clean red team report means the team didn't find problems -- not that problems don't exist.
- **"Automated red teaming replaces human red teamers."** Automated approaches scale well but tend to find known categories of vulnerability. Novel, creative attack vectors are still primarily discovered by humans. The most effective programs combine both.
- **"Red teaming is a one-time activity."** New attack techniques are constantly being discovered, models are continually updated, and the threat landscape evolves. Red teaming must be ongoing.
- **"Only security experts can red team."** While security expertise is valuable, some of the most important findings come from domain experts (a doctor finding dangerous medical advice) or from people with diverse cultural backgrounds identifying region-specific harms.
- **"Red teaming is just trying to make the model say bad words."** Sophisticated red teaming goes far beyond surface-level content policy violations to test for subtle biases, reasoning errors in high-stakes domains, privacy vulnerabilities, and systemic failure modes.

## Connections to Other Concepts

- **Prompt Injection & Jailbreaking**: These are specific attack categories within the broader red teaming exercise. Red teaming provides the structured framework for testing these and other vulnerabilities.
- **Guardrails & Content Filtering**: Red teaming directly evaluates the effectiveness of guardrails and identifies gaps where filtering fails.
- **Bias & Fairness**: Bias testing is a core component of red teaming, and red team exercises often uncover biases that automated fairness metrics miss.
- **The Alignment Problem**: Red teaming is a practical, empirical approach to discovering alignment failures -- cases where the model's behavior diverges from what its developers intended.
- **RLHF**: Red team findings are often used to generate additional training data for RLHF, creating a feedback loop between adversarial testing and model improvement.

## Diagrams and Visualizations

*Recommended visual: Red teaming pipeline: define scope, design attacks, execute probes, document findings, remediate — see [Anthropic Red Teaming Research](https://www.anthropic.com/research)*

*Recommended visual: Automated red teaming with attacker LLM generating adversarial prompts — see [Perez et al. Red Teaming Paper (arXiv:2202.03286)](https://arxiv.org/abs/2202.03286)*

## Further Reading

- Ganguli et al., "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned" (2022) -- Anthropic's comprehensive study on red teaming methodology and findings at scale.
- Perez et al., "Red Teaming Language Models with Language Models" (2022) -- Foundational work on using LLMs to automatically generate adversarial test cases for other LLMs.
- Mazeika et al., "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal" (2024) -- Introduces the HarmBench framework for standardized, reproducible evaluation of red teaming attacks and defenses.

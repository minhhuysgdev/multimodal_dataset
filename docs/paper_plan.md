# Do Hierarchies Help? Minimal Intent Prototypes for Vietnamese Hierarchical Intent Retrieval

## Abstract

Retrieval-based intent classification is a practical approach for reducing the search space in hierarchical intent taxonomies. Instead of directly predicting an intent label from a large label set, the system first retrieves a small set of candidate intents and then performs downstream classification or annotation within that narrowed space. However, a key design question remains unclear: how should intent labels be represented for dense retrieval?

A common intuition is to enrich each intent representation with hierarchy-derived context such as taxonomy paths or higher-level category information. In this work, we revisit this assumption in the context of Vietnamese e-commerce queries, where user inputs are often short, noisy, and domain-specific. We propose a **Minimal Intent Prototype Representation**, where each intent is represented only by its label, natural-language description, detection signals, and example utterances. Unlike hierarchy-augmented representations, our main representation does not inject additional taxonomy context into the dense retrieval text.

We construct a Vietnamese hierarchical intent retrieval benchmark from e-commerce customer queries and a three-level L1–L2–L3 intent taxonomy. We compare minimal prototypes with lightweight hierarchy-aware variants across multiple embedding models and retrieval settings. Beyond retrieval accuracy, the prototype-based formulation also supports **training-free taxonomy extension**: new intents can be added by defining and encoding new prototypes without retraining the full model. This makes the approach suitable for evolving domain-specific intent taxonomies.

***

## 1. Introduction

Hierarchical intent classification is widely used in practical systems such as customer support automation, FAQ retrieval, and conversational agents. In these systems, user queries are mapped to a structured taxonomy of intents. The taxonomy is often organized into multiple levels, such as high-level domains, functional categories, and fine-grained intent labels.

This structure is useful because it organizes the label space and makes the system easier to inspect. However, it also introduces challenges. As the taxonomy grows, direct classification over the entire label set becomes more difficult. Fine-grained intents may be semantically close, and short user queries may not provide enough information for reliable classification.

A common solution is retrieval-based intent classification. Instead of predicting a label directly from the full taxonomy, the system first retrieves a small set of candidate intents. A downstream classifier, LLM annotator, or human reviewer then selects the final label from this candidate set. This approach reduces the search space and improves interpretability.

With sentence embedding models, retrieval can be implemented by encoding both the input query and intent representations into a shared vector space. Sentence embedding approaches make semantic search practical by producing vector representations that can be compared using similarity measures such as cosine similarity.

However, a key design question remains: **how should intent labels be represented for dense retrieval?** A natural intuition is to enrich each intent representation with hierarchy-derived information, such as taxonomy paths, so that the representation captures not only the intent itself but also its position in the taxonomy.

In this paper, we question this assumption. While hierarchy-derived context can be useful for validation and analysis, injecting it directly into the dense retrieval text may introduce noise or reduce the specificity of the target intent representation. A taxonomy path can provide useful coarse context, but it can also make the prototype less focused on the fine-grained meaning of the intent.

We focus on Vietnamese e-commerce queries, where user inputs are often short, informal, and domain-specific. This setting makes representation design especially important. We propose a **Minimal Intent Prototype Representation**, where each intent is represented using only the information that directly describes its meaning: label, description, detection signals, and example utterances.

Our main hypothesis is simple:

> For dense intent retrieval, a compact semantic prototype can be more reliable than a verbose hierarchy-augmented prototype.

We construct a Vietnamese hierarchical intent retrieval benchmark and compare minimal prototypes with lightweight hierarchy-aware variants. We also compare multiple embedding models, including lightweight multilingual encoders and Vietnamese-oriented retrieval models.

Our contributions are fourfold:

1. We introduce a Vietnamese e-commerce hierarchical intent retrieval benchmark based on customer queries and a three-level L1–L2–L3 intent taxonomy.
2. We propose **Minimal Intent Prototype Representation**, a compact semantic representation for dense intent retrieval.
3. We conduct a controlled ablation study comparing minimal prototypes with hierarchy-aware variants.
4. We show that prototype-based retrieval naturally supports **training-free taxonomy extension**, allowing new intents to be added without retraining the full model.

Overall, this work reframes hierarchical intent retrieval as a representation design problem rather than a model architecture problem.

***

## 2. Dataset

We construct a Vietnamese e-commerce intent dataset from customer question–answer data and product-related text. The dataset contains user queries related to product information, usage, pricing, suitability, purchase decisions, and after-sale support.

Each query is mapped to a hierarchical intent taxonomy with three levels:

* **L1**: high-level domain or customer journey stage
* **L2**: functional intent category
* **L3**: fine-grained intent label

For example, a query such as:

```text
Laptop này pin dùng được mấy tiếng?
```

may be mapped to:

```text
L1: truoc_mua_hang
L2: laptop_pin
L3: laptop_pin_thoi_luong
```

Each intent node contains metadata such as:

* intent label
* natural-language description
* detection signals
* example utterances

The taxonomy itself is stored as a structured graph with parent–child relationships. However, in our main retrieval method, only the semantic prototype of each intent is used for dense retrieval. The taxonomy structure is retained for validation, evaluation, and error analysis.

The benchmark contains queries paired with their gold L1–L2–L3 intent paths. This allows us to evaluate both retrieval quality and downstream label accuracy.

***

## 3. Related Work

### 3.1 Sentence Embeddings and Semantic Retrieval

Sentence embedding models are widely used for semantic search and retrieval. These models encode text into dense vectors, allowing queries and candidate representations to be compared in the same vector space.

Our work uses this general retrieval principle: both user queries and intent prototypes are encoded into dense vectors, and retrieval is performed using vector similarity.

### 3.2 Hierarchical Text Classification

Hierarchical text classification studies how to classify documents or utterances into a structured label hierarchy. Prior work has explored ways to incorporate label hierarchy into model architectures, training objectives, or prediction constraints.

However, these methods usually focus on supervised classification. In contrast, our work focuses on a different question: when using sentence embedding models for retrieval, should hierarchy-derived context be injected into the textual representation of each label?

This distinction is important because dense retrieval models encode semantic similarity, not necessarily hierarchical decision boundaries.

### 3.3 Vietnamese Embedding Models

Vietnamese semantic retrieval requires encoders that can handle Vietnamese language patterns and domain-specific expressions. Multilingual encoders provide a useful baseline, but Vietnamese-oriented retrieval models may better capture local phrasing, short queries, and domain-specific terms.

This motivates our encoder comparison in the experimental setup.

### 3.4 LLM-Assisted Annotation

LLMs are increasingly used to support data annotation. In our pipeline, candidate retrieval narrows the possible intent labels before an LLM annotator predicts the final label.

This motivates treating retrieval as a critical upstream component. If the correct intent is not retrieved, the downstream annotator is unlikely to produce the correct final label.

***

## 4. Method

### 4.1 Problem Formulation

Given a user query $$q$$, the goal is to retrieve a set of Top-K candidate intents from an intent taxonomy:

$$
C_K(q) = \text{TopK}_i \; sim(q, i)
$$

where $$sim(q, i)$$ measures the similarity between query $$q$$ and intent $$i$$.

The retrieval module does not directly produce the final label. Instead, it returns a candidate set that can be consumed by a downstream classifier, LLM annotator, or human reviewer.

The goal is to maximize the probability that the gold L3 intent appears in the Top-K candidate set and appears as high as possible in the ranking.

***

### 4.2 Minimal Intent Prototype Representation

For each intent $$i$$, we construct a textual prototype $$p_i$$. The proposed **Minimal Intent Prototype Representation** includes only information directly describing the target intent:

* intent label
* natural-language description
* detection signals
* example utterances

Formally:

$$
p_i = concat(label_i, description_i, signals_i, examples_i)
$$

For example, for the intent:

```text
truoc_mua_hang.laptop_pin.laptop_pin_thoi_luong
```

the minimal prototype is:

```text
Intent: laptop_pin_thoi_luong
Description: User asks about laptop battery duration after a full charge.
Detection signals: pin, thời lượng pin, dùng được bao lâu, mấy tiếng
Example: Laptop pin được bao lâu?
```

The prototype does **not** include additional taxonomy context in the main retrieval representation.

This is a deliberate design choice. Additional taxonomy context may provide coarse structural information, but it may also reduce the specificity of the target intent representation in dense embedding space.

***

### 4.3 Prototype Encoding

Each prototype is encoded using a sentence embedding model:

$$
h_i = Enc(p_i)
$$

where $$Enc(\cdot)$$ is the encoder and $$h_i$$ is the dense vector representation of intent $$i$$.

All prototype vectors are stored in an intent prototype memory:

$$
H = \{h_1, h_2, ..., h_n\}
$$

The query is encoded using the same encoder:

$$
h_q = Enc(q)
$$

Similarity is computed using cosine similarity:

$$
sim(q, i) = cos(h_q, h_i)
$$

***

### 4.4 Candidate Retrieval

The Top-K candidate intents are retrieved as:

$$
C_K(q) = TopK_i \; cos(h_q, h_i)
$$

This retrieval step can be used alone or combined with keyword-based matching. In our system, semantic retrieval can be combined with regex-based retrieval over detection signals:

$$
C_K^{union}(q) = dedup(C_K^{semantic}(q) \cup C_K^{regex}(q))
$$

This union strategy is useful because Vietnamese e-commerce queries often contain clear lexical signals such as:

```text
giá
pin
đổi trả
bảo hành
da dầu
```

***

### 4.5 Hierarchy-Aware Variant

Although our main representation is minimal, we evaluate a lightweight hierarchy-aware variant as an ablation:

| Representation      | Prototype Content                        |
| ------------------- | ---------------------------------------- |
| Label-only          | label                                    |
| Label + Description | label + description                      |
| Minimal Prototype   | label + description + signals + examples |
| Minimal + Path      | minimal + taxonomy path                  |

The purpose of this ablation is to test whether adding taxonomy path information improves or harms dense retrieval.

***

### 4.6 Role of the Taxonomy Graph

The taxonomy graph is not discarded. It is used for:

1. **Taxonomy validation**  
   Ensuring that predicted L1–L2–L3 paths exist.

2. **Hierarchical evaluation**  
   Computing L1, L2, L3, and path-level metrics.

3. **Error analysis**  
   Distinguishing near errors from far errors.

4. **Candidate organization**  
   Presenting candidates in a structured way for annotation or review.

The key distinction is that taxonomy structure is used for validation and analysis, but not directly injected into the main dense prototype representation.

***

### 4.7 Training-Free Taxonomy Extension

A practical advantage of prototype-based retrieval is that it separates taxonomy updates from model training. In conventional supervised intent classifiers, adding a new intent changes the output label space and often requires retraining or fine-tuning the classifier.

In contrast, our method represents each intent as an independent textual prototype. When a new intent appears, the system only needs to define its label, description, detection signals, and example utterances:

$$
p_{new} = concat(label_{new}, description_{new}, signals_{new}, examples_{new})
$$

The new prototype is encoded using the existing encoder:

$$
h_{new} = Enc(p_{new})
$$

Then the intent memory is updated:

$$
H' = H \cup \{h_{new}\}
$$

No classifier parameters are updated, and the full model does not need to be retrained. This makes the framework suitable for evolving e-commerce taxonomies, where new intents may appear due to new products, policy changes, campaigns, or newly observed customer concerns.

We refer to this property as **training-free taxonomy extension**, rather than continual learning in the strict model-training sense. Adaptation happens at the prototype memory level, not through parameter updates.

A concise summary is:

> The system does not learn new intents by updating model parameters; it adapts by extending the prototype memory.

***

## 5. Experimental Setup

### 5.1 Representation Baselines

We compare the following prototype representations:

| Representation      | Content                                  |
| ------------------- | ---------------------------------------- |
| Label-only          | label                                    |
| Label + Description | label + description                      |
| Minimal Prototype   | label + description + signals + examples |
| Minimal + Path      | minimal + taxonomy path                  |

***

### 5.2 Encoder Baselines

We compare different embedding models:

| Encoder               | Role                                     |
| --------------------- | ---------------------------------------- |
| MiniLM                | lightweight multilingual baseline        |
| BGE-M3                | strong multilingual retrieval baseline   |
| Vietnamese\_Embedding | Vietnamese-oriented retrieval encoder    |
| multilingual E5       | retrieval-oriented multilingual baseline |

***

### 5.3 Retrieval Strategies

We evaluate:

| Strategy        | Description                            |
| --------------- | -------------------------------------- |
| Regex-only      | keyword and detection-signal matching  |
| Semantic-only   | dense retrieval over prototypes        |
| Union retrieval | regex candidates ∪ semantic candidates |

***

### 5.4 Metrics

We evaluate retrieval using:

* **Recall\@K**: whether the correct intent appears in the Top-K candidates
* **MRR\@K**: how early the correct intent appears in the ranking
* **nDCG\@K**: ranking quality with hierarchy-aware graded relevance

For downstream labeling, we report:

* L1 accuracy
* L2 accuracy
* L3 accuracy
* full-path accuracy
* Macro-F1 at L3

For taxonomy extension, we report:

* new-intent Recall\@K
* new-intent MRR\@K
* existing-intent Recall\@K before and after insertion
* interference rate on existing intents

***

## 6. Analysis

### 6.1 Why Can Additional Taxonomy Context Hurt Dense Retrieval?

Although taxonomy structure is useful, injecting taxonomy-derived text directly into dense retrieval representations can introduce several issues.

First, taxonomy paths provide coarse contextual information, but they may reduce the specificity of the target intent representation. For example, a high-level path may describe the general topic, while the retrieval task requires distinguishing a fine-grained intent.

Second, dense embedding models are trained to capture semantic similarity, not necessarily to separate fine-grained class boundaries. Adding additional taxonomy context may make prototypes more similar to nearby classes and less focused on the target intent.

Third, dense retrieval works best when the representation clearly describes the semantic meaning of the item being retrieved. A compact prototype containing the intent meaning, detection signals, and examples may therefore be more effective than a verbose representation with additional structural context.

***

### 6.2 Why Minimal Prototypes Are Practical

Minimal prototypes are easy to write, inspect, and update. They directly describe what the intent means without requiring the designer to decide how much taxonomy context to include.

This is especially useful in low-resource settings, where there may not be enough labeled data to train a supervised classifier. The system can rely on prototype construction and embedding-based retrieval instead of full model training.

***

### 6.3 Taxonomy Extension without Retraining

The prototype formulation also supports incremental taxonomy updates. When a new intent is added, the system creates a new prototype and inserts its vector into memory. This avoids retraining a classifier whenever the label set changes.

This property is useful in e-commerce, where customer concerns evolve over time. New product types, new promotions, new return policies, or unexpected user questions can introduce new intents.

***

## 7. Conclusion

This paper studies a simple but important question in retrieval-based hierarchical intent classification: how should intents be represented for dense retrieval?

We propose Minimal Intent Prototype Representation, where each intent is represented using only its label, description, detection signals, and example utterances. Unlike hierarchy-augmented representations, the proposed representation does not inject additional taxonomy context into the dense retrieval text.

Our study frames hierarchical intent retrieval as a representation design problem rather than a model architecture problem. Through controlled ablations, we compare minimal prototypes with a lightweight hierarchy-aware variant and analyze whether taxonomy path information helps or hurts retrieval.

Beyond retrieval performance, the prototype-based formulation also supports training-free taxonomy extension. New intents can be added by creating and encoding new prototypes, without retraining the full model.

Overall, the findings suggest that compact semantic prototypes are a strong and practical representation for Vietnamese hierarchical intent retrieval, especially in domain-specific and evolving taxonomy settings.

***

## Final Paper Spine

```text
We show that, for Vietnamese hierarchical intent retrieval, compact intent prototypes based on direct semantic descriptions are a reliable and extensible alternative to dense retrieval representations that inject additional taxonomy context.
```

***

## Final Contribution Paragraph

```text
Our contributions are fourfold. First, we introduce a Vietnamese e-commerce hierarchical intent retrieval benchmark based on real-world customer queries and a three-level L1–L2–L3 taxonomy. Second, we propose Minimal Intent Prototype Representation, a compact semantic representation that encodes each intent using only its label, description, detection signals, and example utterances. Third, we conduct a controlled ablation study comparing minimal prototypes with a lightweight hierarchy-aware variant across multiple embedding models and retrieval strategies. Fourth, we show that the prototype-based formulation supports training-free taxonomy extension, allowing new intents to be added by encoding new prototypes without retraining the full model.
```

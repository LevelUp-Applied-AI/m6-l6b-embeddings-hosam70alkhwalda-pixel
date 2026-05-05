# Stretch 6B-S1 — Embedding Space Explorer: Analysis

## Methodology

### Dimensionality Reduction Choice: t-SNE

I chose **t-SNE** (t-Distributed Stochastic Neighbor Embedding) for both the word-level and document-level visualizations. t-SNE preserves **local neighborhood structure**, pulling semantically similar items into tight, visually distinct clusters — making it far more interpretable than PCA for revealing how embeddings organize meaning.

- **Word embeddings:** `perplexity=30` — a standard value for ~200 points, balancing tight intra-category cohesion with visible separation across categories.
- **Document embeddings:** `perplexity=5` — required because t-SNE's perplexity must be less than the number of samples (20 docs); this low value emphasizes local groupings among articles that share topic vocabulary.

---

## Word Embedding Visualization Analysis

The t-SNE projection of 197 GloVe 50d vectors reveals **five well-separated clusters**, each corresponding to a semantic category, with some meaningful boundary effects:

**Countries** (red) form the most compact and spatially isolated cluster, centered around x=−5, y=−2. Words like `france`, `brazil`, `australia`, `nigeria`, and `china` sit tightly together. This strong cohesion reflects the fact that country names appear in nearly identical syntactic contexts in GloVe's training corpus (geopolitical reporting, Wikipedia infoboxes), giving them very similar distributional signatures.

**Sports** (teal/green) occupy the bottom-left region and show the widest spatial spread of any category. Competition-level terms like `olympics`, `championship`, and `marathon` cluster on the upper-left edge of the group, while the bulk of match-mechanics words spread further down and right. Notably, `marathon` drifts away from the core Sports cluster — consistent with its frequent non-sports usage ("marathon negotiations", "marathon session") in news corpora, which pulls its GloVe vector away from pure sports vocabulary.

**Technology** (blue) occupies the upper-center region as a dense, well-defined cluster. `algorithm` and `smartphone` appear as annotated anchors near the cluster center, while `artificial` sits at the lower boundary of the group — semantically ambiguous as a standalone word (not paired with "intelligence"), its vector is pulled slightly away from the core tech cluster.

**Finance** (orange) forms a large, somewhat dispersed cluster in the upper-right and center-right. Macro-economic terms like `inflation` and `recession` group together on the inner edge, while `nasdaq` drifts to the far right as a notable outlier. This spread reflects Finance vocabulary appearing across diverse editorial contexts (business reporting, political coverage, economics commentary), giving its words a broader distributional range than the tighter Countries cluster.

**Emotions** (purple) occupy the right-center and lower-right, with noticeable internal spread. Basic emotions cluster more tightly at the core, while complex emotional states like `euphoria`, `nostalgia`, and `melancholy` sit at the periphery. High-frequency basic emotion words have tightly learned vectors from repetitive direct contexts, while nuanced emotional vocabulary appears in more varied literary and psychological settings, diffusing their representations in embedding space.

**Key boundary observation:** Technology and Finance clusters are the closest pair in the projection, sharing a soft boundary in the center of the plot. This adjacency is semantically meaningful — financial journalism about tech companies creates cross-domain co-occurrences in GloVe's training data, blurring the vector boundary between these two domains.

---

## Document Embedding Visualization Analysis

The t-SNE projection of 20 BBC News DistilBERT embeddings (768d → 2D) shows **clear topic-level separation**, with each category forming a recognizable region despite the small sample size:

**Sport** (blue) articles cluster tightly in the upper-left region. Articles like "sydney return for henin-hardenne", "republic to face china and italy", "wolves appoint hoddle as manager", and "santini resigns as spurs manager" all group together, showing that DistilBERT's contextual embeddings strongly distinguish sports reporting from other topics. The tight clustering reflects Sport's distinctive vocabulary (player names, match results, managerial news) that creates large angular distances from other categories in 768d space.

**Entertainment** (orange) articles appear in the upper-center region as a loose cluster. "duran duran show set for us tv", "portishead back after eight years", and "angels favourite funeral song" sit in proximity, though with more spread than Sport. This wider scatter reflects Entertainment journalism's genre diversity — music, TV, and celebrity coverage embed differently depending on their sub-topic even within the same BBC category.

**Politics** (teal) articles form a scattered group in the center-left. "cabinet anger at brown cash raid", "blunkett unveils policing plans", "former ni minister scott dies", and "faith schools citizenship warning" span a wider spatial range than Sport, consistent with political reporting covering diverse policy domains (economic policy, security, education) that pull articles toward different neighboring categories.

**Business** (red) articles cluster on the right side of the plot. "india calls for fair trade rules", "us trade gap ballooned in october", "card fraudsters targeting web", and "low-cost airlines hit eurotunnel" group together in the right-center area. Two business articles drift slightly toward the center — likely those with economic policy framing that overlaps with Politics vocabulary.

**Tech** (purple) articles cluster in the lower-right, clearly isolated from the other categories. "halo 2 heralds traffic explosion", "ipods not media players yet", "bond game fails to shake or stir", and "sony psp console hits us in march" sit together as a tight group, confirming that DistilBERT captures product-specific and consumer technology language as a distinct semantic region.

**Overall pattern:** DistilBERT's 768-dimensional embeddings produce cleaner inter-category separation at the document level than GloVe's 50d vectors do at the word level. The model's attention mechanism captures full-sentence context, allowing it to distinguish a tech article about a gaming console from a business article about trade policy — a distinction word-level embeddings would struggle to encode.

---

## Summary

Both visualizations confirm that embedding spaces encode genuine semantic structure. GloVe word vectors cluster by semantic category in 50 dimensions, with border effects between Technology and Finance revealing real-world thematic overlap in training corpora. DistilBERT document embeddings show strong topic-level separation across all five BBC categories, with Sport producing the tightest cluster and Entertainment the most dispersed — reflecting differences in genre diversity within each news category. t-SNE successfully makes these high-dimensional structures directly interpretable, demonstrating its value as a diagnostic tool for understanding and communicating model behavior in production NLP systems.
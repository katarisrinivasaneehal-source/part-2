const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak,
} = require("docx");
const fs = require("fs");

const FIG = "/home/claude/smarthire/reports/figures";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function image(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ data: fs.readFileSync(path), transformation: { width, height }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
  });
}

function makeTable(headers, rows, colWidths) {
  const totalWidth = 9000;
  const widths = colWidths || headers.map(() => totalWidth / headers.length);
  const headerRow = new TableRow({
    children: headers.map((hText, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "2563EB" },
      children: [new Paragraph({ children: [new TextRun({ text: hText, bold: true, color: "FFFFFF", size: 20 })] })],
    })),
  });
  const bodyRows = rows.map((row) => new TableRow({
    children: row.map((cellText, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      children: [new Paragraph({ children: [new TextRun({ text: String(cellText), size: 20 })] })],
    })),
  }));
  return new Table({ width: { size: totalWidth, type: WidthType.DXA }, columnWidths: widths, rows: [headerRow, ...bodyRows] });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "SmartHire", bold: true, size: 56, color: "1E3A8A" })],
        spacing: { after: 80 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Resume-to-Job Matching & Career Guidance Engine", size: 30, color: "444444" })],
        spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Final Project Report — Classical Machine Learning (No LLMs)", size: 24, italics: true, color: "666666" })],
        spacing: { after: 400 },
      }),

      h1("1. Project Overview"),
      p("SmartHire is a portal where a candidate uploads a resume and receives: (a) a predicted job "
        + "category, (b) a ranked list of matching jobs from a merged job corpus, (c) an optional "
        + "fit/shortlisting score for each match, and (d) a skill-gap report highlighting what to "
        + "improve. The entire system is built with classical machine learning — TF-IDF, cosine "
        + "similarity, K-Means, and standard scikit-learn classifiers — with no LLMs or generative AI "
        + "anywhere in the pipeline."),
      p("The system maps onto six ML tasks spanning both supervised and unsupervised learning:"),
      bullet("Resume category classification (supervised) — Model A"),
      bullet("Content-based job recommendation via cosine similarity (unsupervised) — core engine"),
      bullet("Job / role clustering via K-Means (unsupervised)"),
      bullet("Skill-gap detection built on top of the clustering output (unsupervised)"),
      bullet("Shortlisting / fit prediction (supervised, optional) — Model B"),
      bullet("A Streamlit portal tying all of the above together"),

      h1("2. Data"),
      p("Three public datasets were merged into the system:"),
      makeTable(
        ["Dataset", "Rows", "Role"],
        [
          ["Resume Dataset (Kaggle, Snehaan Bhawal)", "2,483 (cleaned)", "Training data for Model A — 24 job categories"],
          ["Naukri Job Listings", "~30,000", "India-focused postings, merged into the job corpus"],
          ["LinkedIn Job Postings 2023–2024", "~40,000 (sampled from 123k)", "Global postings incl. skills + salary, merged into the job corpus"],
        ],
        [4500, 2000, 2500]
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      p("The two job sources were cleaned and merged into a single job corpus with common columns: "
        + "title, company, location, skills, description, experience, source. Neither source had a "
        + "clean, ready-made skills/experience field — Naukri's fields were regex-extracted from the "
        + "free-text description, and LinkedIn's skills came from a separate job_skills.csv keyed by "
        + "job_id, joined against a skill-name mapping table. The merged corpus totals 69,991 postings."),

      h2("2.1 Exploratory Data Analysis"),
      image(`${FIG}/eda_resume_category_dist.png`, 500, 300),
      caption("Figure 1. Resume category distribution — fairly balanced (~100–120 per class), "
        + "except Automobile and Bpo which are noticeably smaller."),
      image(`${FIG}/eda_job_corpus_source.png`, 280, 280),
      caption("Figure 2. Job corpus composition by source after merging and cleaning."),

      new Paragraph({ children: [new PageBreak()] }),

      h1("3. Model A — Resume Category Classifier (Supervised)"),
      p("Pipeline: clean resume text → TF-IDF (8,000 features, 1–2 grams) → multi-class classifier. "
        + "Three models were trained and compared with class-balanced weighting to handle the mild "
        + "class imbalance (Automobile: 36 resumes, Information-Technology: 120 resumes):"),
      makeTable(
        ["Model", "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
        [
          ["Logistic Regression", "0.690", "0.671", "0.651", "0.647"],
          ["Linear SVM", "0.718", "0.702", "0.680", "0.675"],
          ["Random Forest (selected)", "0.765", "0.799", "0.731", "0.734"],
        ],
        [3000, 1500, 1750, 1500, 1250]
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      p("Random Forest was selected as the best model by macro F1. It reaches 76.5% accuracy across "
        + "24 classes (random guessing ≈ 4%)."),
      image(`${FIG}/confusion_matrix_classifier.png`, 480, 400),
      caption("Figure 3. Confusion matrix, Random Forest classifier on the held-out test set. Strong "
        + "diagonal; most confusion is between semantically adjacent categories (e.g. Arts vs. "
        + "Digital-Media, Consultant vs. Finance)."),
      p("Limitations: Automobile and Bpo have the fewest training examples, which shows up as lower "
        + "recall for those classes. Confusable categories share overlapping vocabulary, which is the "
        + "main source of remaining misclassification."),

      h1("4. Job Recommender (Unsupervised Core Engine)"),
      p("The full job corpus (69,991 postings) is vectorized once with TF-IDF (15,000 features, "
        + "1–2 grams). For any resume, the same vectorizer transforms the resume text into the same "
        + "space, and jobs are ranked by cosine similarity — this is the heart of the portal's "
        + "'search' feature."),
      p("Qualitatively, recommendations are strong: an HR resume's top-5 matches are all HR/People-"
        + "Ops management roles; a Python/Django/AWS engineer's top matches are DevOps and Python "
        + "developer roles. A quantitative Precision@10 check was also run using a keyword-overlap "
        + "proxy (does the resume's category name share a token with the recommended job's title) "
        + "since the job corpus has no ground-truth category label. This proxy is intentionally "
        + "strict and undercounts true matches (e.g. category \"Hr\" vs. title \"Human Resources "
        + "Manager\" share no exact token), so it should be read as a conservative lower bound rather "
        + "than an absolute score — the qualitative spot-checks give a fuller picture."),

      h1("5. Job / Role Clustering (Unsupervised)"),
      p("The job TF-IDF matrix was reduced to 100 dimensions via TruncatedSVD (16.3% explained "
        + "variance), then K-Means was run for k ∈ {4, 6, ..., 20}. k was selected by silhouette "
        + "score (computed on an 8,000-row subsample for speed), with k=20 scoring highest."),
      image(`${FIG}/clustering_k_selection.png`, 500, 210),
      caption("Figure 4. Elbow (inertia) and silhouette score across candidate k values."),
      image(`${FIG}/clustering_pca_scatter.png`, 420, 330),
      caption("Figure 5. PCA projection of the 20 job clusters. Visible structure with expected "
        + "overlap from projecting 20 clusters down to 2 dimensions."),
      p("The resulting clusters are interpretable role families: nursing/healthcare, Java development, "
        + "web development, marketing, accounting, BPO/call-center, sales/retail, and several general "
        + "clusters. Silhouette scores are low overall (~0.03–0.05), typical for sparse, high-"
        + "dimensional TF-IDF text with generic boilerplate language (EEO clauses, generic "
        + "\"requirements\" phrasing) diluting cohesion — a few clusters are dominated by this "
        + "boilerplate rather than job-specific content, which is a known limitation."),

      h1("6. Skill-Gap Report (Unsupervised Insight Module)"),
      p("For a candidate resume, the system predicts its nearest job cluster (via the fitted SVD + "
        + "K-Means models), extracts that cluster's most common skills from a curated ~100-term skill "
        + "vocabulary matched against job text, and compares it against the candidate's own extracted "
        + "skills to report matched vs. missing skills. This directly powers the \"CV preparation\" "
        + "feature from the original brief."),
      p("This module gives directionally sensible advice — e.g. a Python/Django engineer resume gets "
        + "shown adjacent missing skills like Java, JavaScript, and Azure. Its main limitation is "
        + "coverage: skills outside the curated vocabulary are invisible to both candidate-skill "
        + "extraction and cluster-skill extraction."),

      new Paragraph({ children: [new PageBreak()] }),

      h1("7. Model B — Fit / Shortlisting Predictor (Supervised, Optional)"),
      p("There is no public ground-truth \"was this candidate shortlisted\" dataset, so training "
        + "labels are heuristic: a (resume, job) pair is labeled \"fit\" if the resume's known "
        + "category shares a keyword with the job title, otherwise \"not fit.\" 1,440 training pairs "
        + "were generated this way across 180 sampled resumes."),
      p("Features: skill overlap (count + Jaccard), resume/job skill counts, an education-keyword "
        + "match flag, job-side required experience (years), and text similarity (cosine similarity "
        + "in the job TF-IDF space)."),
      makeTable(
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        [
          ["Logistic Regression (selected)", "0.689", "0.707", "0.644", "0.674", "0.768"],
          ["XGBoost", "0.678", "0.708", "0.606", "0.653", "0.750"],
        ],
        [3200, 1300, 1300, 1200, 1200, 1300]
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      image(`${FIG}/fit_predictor_roc.png`, 350, 300),
      caption("Figure 6. ROC curves, Logistic Regression vs. XGBoost fit predictor."),
      p("Logistic Regression slightly outperformed XGBoost — with only 7 engineered features and a "
        + "noisy heuristic label, a simpler linear model generalizes marginally better than a tree "
        + "ensemble prone to overfitting small feature sets."),
      p("Important caveat: text_similarity is expected to be the single strongest feature, since it's "
        + "closely related to how the heuristic labels were constructed (title keyword overlap "
        + "correlates with topical similarity) — this is a known form of label leakage, flagged here "
        + "rather than hidden. In live testing through the Streamlit app, fit scores for close job "
        + "matches came out very close to 1.0, suggesting the model is somewhat overconfident / poorly "
        + "calibrated. This module should be read as a proof-of-concept demonstrating the full "
        + "supervised workflow (feature engineering, class-balance handling, model comparison, "
        + "ROC-AUC), not as a production-ready shortlisting tool. A real deployment would need an "
        + "actual labeled shortlisting dataset, e.g. from ATS decisions."),

      h1("8. Streamlit Portal"),
      p("All modules are tied together in app/streamlit_app.py. A user uploads a resume (PDF, DOCX, "
        + "or TXT) or pastes text; the app extracts and cleans the text, then displays:"),
      bullet("The predicted job category (Model A)"),
      bullet("A ranked table of top-N matching jobs with cosine similarity match scores"),
      bullet("A fit/shortlist probability per matched job (Model B, when available)"),
      bullet("A skill-gap report: detected skills, matched skills, and skills to develop"),
      p("Model and job-corpus artifacts are cached with @st.cache_resource so the app stays responsive "
        + "after the first load. The app was smoke-tested end-to-end with real resume text and confirmed "
        + "to boot and return sensible results for multiple resume categories."),

      h1("9. Limitations & Future Work"),
      bullet("The job corpus caps each source at 40,000 rows for speed (MAX_JOBS_PER_SOURCE in "
        + "src/config.py) — raising this, or moving to MiniBatch/approximate methods throughout, "
        + "would let the full ~150k+ postings be used."),
      bullet("Several clusters are dominated by boilerplate legal/EEO language rather than job "
        + "content; more aggressive boilerplate stripping before vectorization would sharpen cluster "
        + "quality."),
      bullet("The fit predictor's heuristic labels are a stand-in for real shortlisting outcomes; "
        + "replacing them with actual ATS decision data (if available) would substantially improve "
        + "the model's validity and calibration."),
      bullet("The skill-gap module's vocabulary (~100 curated terms) is a coverage bottleneck — an "
        + "expanded or learned skill taxonomy would catch more nuanced or emerging skills."),
      bullet("Stretch goals not attempted in this pass: sentence-embedding-based matching (still "
        + "classical ML in spirit, but more expensive), a learning-to-rank model for job ordering, "
        + "and public deployment on Streamlit Community Cloud."),

      h1("10. Conclusion"),
      p("SmartHire delivers the full minimum-scope requirement — a resume classifier, a content-based "
        + "job recommender, and a skill-gap report — plus the optional fit/shortlisting predictor, all "
        + "using classical ML techniques and real-world datasets. The strongest components are the "
        + "resume classifier (76.5% accuracy across 24 classes) and the job recommender (qualitatively "
        + "strong, fast, and interpretable); the fit predictor is the weakest link due to the absence "
        + "of real shortlisting ground truth, and is presented as a demonstration of the supervised "
        + "workflow rather than a production-ready feature."),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("/home/claude/smarthire/reports/final_report.docx", buffer);
  console.log("Report written.");
});

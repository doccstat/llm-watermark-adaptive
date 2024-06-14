set.seed(1)

folder <- "results/"
models <- c("facebook/opt-1.3b", "openai-community/gpt2")
models_folders_prefix <- c("opt", "gpt")
generation_methods <- c("gumbel")
attacks <- c("deletion", "insertion", "substitution")
n <- 20
m <- 20
attack_pcts <- c(
  "0.0", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"
)
watermarked_or_null <- c("watermarked")

pvalue_files_templates <- matrix(NA, 0, 15)
for (model_index in seq_along(models)) {
  for (generation_methods_index in seq_along(generation_methods)) {
    for (attack_index in seq_along(attacks)) {
      for (attack_pcts_index in seq_along(attack_pcts)) {
        for (watermarked_or_null_index in seq_along(watermarked_or_null)) {
          pvalue_files_templates <- rbind(pvalue_files_templates, c(
            folder,
            models_folders_prefix[model_index],
            "-",
            generation_methods[generation_methods_index],
            "-",
            attacks[attack_index],
            "-",
            n,
            "-",
            m,
            "-",
            attack_pcts[attack_pcts_index],
            ".p-detect/",
            watermarked_or_null[watermarked_or_null_index],
            "-XXX.csv"
          ))
        }
      }
    }
  }
}

prompt_count <- 200
df <- matrix(NA, 0, 5 + 5)

filename <- sub("XXX", 0, paste0(pvalue_files_templates[1, ], collapse = ""))
metric_count <- ncol(read.csv(filename, header = FALSE))

for (template_index in seq_len(nrow(pvalue_files_templates))) {
  probs <- read.csv(
    paste0(
      paste0(
        pvalue_files_templates[template_index, seq_len(12)], collapse = ""
      ), ".p-probs.csv"
    ), header = FALSE
  )
  empty_probs <- read.csv(
    paste0(
      paste0(
        pvalue_files_templates[template_index, seq_len(12)], collapse = ""
      ), ".p-empty-probs.csv"
    ), header = FALSE
  )
  for (prompt_index in seq_len(prompt_count)) {
    # Python index in the file name
    filename <- sub(
      "XXX",
      prompt_index - 1,
      paste0(pvalue_files_templates[template_index, ], collapse = "")
    )
    df <- rbind(
      df,
      cbind(
        pvalue_files_templates[template_index, 2],
        pvalue_files_templates[template_index, 4],
        pvalue_files_templates[template_index, 6],
        pvalue_files_templates[template_index, 12],
        prompt_index,
        paste("Metric", seq_len(metric_count)),
        matrix(read.csv(filename, header = FALSE)),
        sum(abs(probs[prompt_index, ] - empty_probs[prompt_index, ])),
        sqrt(sum((probs[prompt_index, ] - empty_probs[prompt_index, ])^2)),
        max(abs(probs[prompt_index, ] - empty_probs[prompt_index, ]))
      )
    )
  }
}

df <- data.frame(df)
names(df) <- c(
  "LLM", "GenerationMethod", "Attack", "AttackPct", "PromptIndex", "Metric",
  "PValue", "ProbsErrorL1Norm", "ProbsErrorL2Norm", "ProbsErrorInfNorm"
)
df <- as.data.frame(lapply(df, unlist))
df$AttackPct <- as.numeric(df$AttackPct)
df$PValue <- as.numeric(df$PValue)
df$ProbsErrorL1Norm <- as.numeric(df$ProbsErrorL1Norm)
df$ProbsErrorL2Norm <- as.numeric(df$ProbsErrorL2Norm)
df$ProbsErrorInfNorm <- as.numeric(df$ProbsErrorInfNorm)

powers <- rbind(
  cbind(
    Threshold = 0.05,
    aggregate(
      df$PValue <= 0.05,
      by = list(
        LLM = df$LLM,
        GenerationMethod = df$GenerationMethod,
        Attack = df$Attack,
        AttackPct = df$AttackPct,
        Metric = df$Metric
      ),
      FUN = mean
    )
  ),
  cbind(
    Threshold = 0.01,
    aggregate(
      df$PValue <= 0.01,
      by = list(
        LLM = df$LLM,
        GenerationMethod = df$GenerationMethod,
        Attack = df$Attack,
        AttackPct = df$AttackPct,
        Metric = df$Metric
      ),
      FUN = mean
    )
  )
)
powers <- data.frame(powers)

p <- ggplot2::ggplot() +
  ggplot2::geom_line(
    ggplot2::aes(x = AttackPct, y = x, color = Metric),
    data = powers[powers$Threshold == 0.05, ]
  ) +
  ggplot2::facet_grid(LLM ~ GenerationMethod + Attack) +
  ggplot2::theme_minimal() +
  ggplot2::scale_x_continuous(labels = scales::percent)
ggplot2::ggsave("results/powers-0.05.pdf", p, width = 10, height = 7)

p <- ggplot2::ggplot() +
  ggplot2::geom_line(
    ggplot2::aes(x = AttackPct, y = x, color = Metric),
    data = powers[powers$Threshold == 0.01, ]
  ) +
  ggplot2::facet_grid(LLM ~ GenerationMethod + Attack) +
  ggplot2::theme_minimal() +
  ggplot2::scale_x_continuous(labels = scales::percent)
ggplot2::ggsave("results/powers-0.01.pdf", p, width = 10, height = 7)

df_probs <- NULL
for (model_prefix in models_folders_prefix) {
  df_llm <- df[df$LLM == model_prefix, ]
  both_metric2_metric3 <- NULL
  for (prompt_index in seq_len(prompt_count)) {
    both_metric2_metric3 <- c(
      both_metric2_metric3,
      df_llm[metric_count * (prompt_index - 1) + 2, "PValue"] <= 0.05 &
        df_llm[metric_count * (prompt_index - 1) + 3, "PValue"] <= 0.05
    )
  }
  both_metric2_metric3 <- seq_len(prompt_count)[both_metric2_metric3]

  metric2_not_metric3 <- NULL
  for (prompt_index in seq_len(prompt_count)) {
    metric2_not_metric3 <- c(
      metric2_not_metric3,
      df_llm[metric_count * (prompt_index - 1) + 2, "PValue"] <= 0.05 &
        df_llm[metric_count * (prompt_index - 1) + 3, "PValue"] > 0.05
    )
  }
  metric2_not_metric3 <- seq_len(prompt_count)[metric2_not_metric3]

  df2 <- df_llm[
    df_llm$PromptIndex %in% both_metric2_metric3 & df_llm$Metric == "Metric 2",
  ]
  df2 <- data.frame(
    LLM = model_prefix,
    PromptIndex = rep(both_metric2_metric3, each = 3),
    Norms = rep(c("L1", "L2", "Inf"), length(both_metric2_metric3)),
    Value = matrix(t(as.matrix(
      df2[, c("ProbsErrorL1Norm", "ProbsErrorL2Norm", "ProbsErrorInfNorm")]
    )), byrow = TRUE),
    group = "Both Metric 2 and Metric 3"
  )
  df3 <- df_llm[
    df_llm$PromptIndex %in% metric2_not_metric3 & df_llm$Metric == "Metric 3",
  ]
  df3 <- data.frame(
    LLM = model_prefix,
    PromptIndex = rep(metric2_not_metric3, each = 3),
    Norms = rep(c("L1", "L2", "Inf"), length(metric2_not_metric3)),
    Value = matrix(t(as.matrix(
      df3[, c("ProbsErrorL1Norm", "ProbsErrorL2Norm", "ProbsErrorInfNorm")]
    )), byrow = TRUE),
    group = "Metric 2 but not Metric 3"
  )
  df_probs <- rbind(df_probs, df2, df3)
}
p <- ggplot2::ggplot() +
  ggplot2::geom_boxplot(
    ggplot2::aes(x = LLM, y = Value, fill = group),
    data = df_probs
  ) +
  ggplot2::theme_minimal() +
  ggplot2::facet_wrap(~ Norms, scales = "free_y") +
  ggplot2::theme(
    strip.text.x = ggplot2::element_text(margin = ggplot2::margin(1, 1, 1, 1))
  ) +
  ggplot2::scale_y_continuous(limits = c(0, NA))
ggplot2::ggsave(
  "results/probs-error-metric23-boxplot.pdf", p, width = 10, height = 6
)

p <- ggplot2::ggplot() +
  ggplot2::geom_histogram(
    ggplot2::aes(x = PValue, fill = Metric),
    data = df,
    bins = 20
  ) +
  ggplot2::facet_wrap(~Metric) +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "none")
ggplot2::ggsave(
  paste0("results/", model_prefix, "-histogram.pdf"), p, width = 7, height = 7
)

comparison_bipartite <- rbind(
  cbind(
    "Metric 1",
    df[df$LLM == model_prefix, "Metric.1"],
    "Metric 3",
    df[df$LLM == model_prefix, "Metric.3"],
    df[df$LLM == model_prefix, "Metric.1"] >=
      df[df$LLM == model_prefix, "Metric.3"]
  ),
  cbind(
    "Metric 3",
    df[df$LLM == model_prefix, "Metric.3"],
    "Metric 2",
    df[df$LLM == model_prefix, "Metric.2"],
    df[df$LLM == model_prefix, "Metric.3"] >=
      df[df$LLM == model_prefix, "Metric.2"]
  )
)
comparison_bipartite <- data.frame(comparison_bipartite)
names(comparison_bipartite) <- c("x", "y", "xend", "yend", "better")
comparison_bipartite <- as.data.frame(lapply(comparison_bipartite, unlist))
comparison_bipartite$y <- as.numeric(comparison_bipartite$y)
comparison_bipartite$yend <- as.numeric(comparison_bipartite$yend)

p <- ggplot2::ggplot(
  comparison_bipartite,
  ggplot2::aes(x = x, y = y, xend = xend, yend = yend, color = better)
) +
  ggplot2::geom_segment() +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "none") +
  ggplot2::scale_y_log10() +
  ggplot2::scale_x_discrete(
    limits = c("Metric 1", "Metric 3", "Metric 2"),
    labels = c("Metric 1", "Metric 3", "Metric 2"),
    expand = c(0, 0.1)
  )
ggplot2::ggsave(
  paste0("results/", model_prefix, "-comparison_bipartite.pdf"),
  p,
  width = 7,
  height = 7
)

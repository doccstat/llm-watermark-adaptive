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
df <- matrix(NA, 0, 5 + 2)

for (template_index in seq_len(nrow(pvalue_files_templates))) {
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
        c("Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5"),
        matrix(read.csv(filename, header = FALSE))
      )
    )
  }
}

df <- data.frame(df)
names(df) <- c(
  "LLM", "GenerationMethod", "Attack", "AttackPct", "PromptIndex",
  "Metric", "PValue"
)
df <- as.data.frame(lapply(df, unlist))
df$AttackPct <- as.numeric(df$AttackPct)
df$PValue <- as.numeric(df$PValue)

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

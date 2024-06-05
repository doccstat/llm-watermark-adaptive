set.seed(1)

folder <- "results/"
models <- c("facebook/opt-1.3b", "openai-community/gpt2")
models_folders_prefix <- c("opt", "gpt")
generation_methods <- c("gumbel")
n <- 10
m <- 10
watermarked_or_null <- c("watermarked")

pvalue_files_templates <- matrix(NA, 0, 11)
for (model_index in seq_along(models)) {
  for (generation_methods_index in seq_along(generation_methods)) {
    for (watermarked_or_null_index in seq_along(watermarked_or_null)) {
      pvalue_files_templates <- rbind(pvalue_files_templates, c(
        folder,
        models_folders_prefix[model_index],
        "-",
        generation_methods,
        "-",
        n,
        "-",
        m,
        ".p-detect/",
        watermarked_or_null[watermarked_or_null_index],
        "-XXX.csv"
      ))
    }
  }
}
# filenames_to_settings <- list(
#   "Setting 1, Gumbel",
#   "Setting 1, Gumbel Edit",
#   "Setting 1, Inverse Transform",
#   "Setting 1, Inverse Transform Edit",
#   "Setting 2, Gumbel",
#   "Setting 2, Gumbel Edit",
#   "Setting 2, Inverse Transform",
#   "Setting 2, Inverse Transform Edit",
#   "Setting 3, Gumbel",
#   "Setting 3, Gumbel Edit",
#   "Setting 3, Inverse Transform",
#   "Setting 3, Inverse Transform Edit",
#   "Setting 4, Gumbel",
#   "Setting 4, Gumbel Edit",
#   "Setting 4, Inverse Transform",
#   "Setting 4, Inverse Transform Edit"
# )
# filenames_to_settings <- rep(filenames_to_settings, length(models))
# names(filenames_to_settings) <- pvalue_files_templates

prompt_count <- 250
df <- matrix(NA, 0, 6)

for (template_index in seq_len(nrow(pvalue_files_templates))) {
  for (prompt_index in seq_len(prompt_count)) {
    # Python index in the file name
    # filename <- sub("XXX", prompt_index - 1, pvalue_files_template)
    filename <- sub(
      "XXX",
      prompt_index,
      paste0(pvalue_files_templates[template_index, ], collapse = "")
    )
    df <- rbind(
      df,
      c(
        pvalue_files_templates[template_index, 2],
        pvalue_files_templates[template_index, 4],
        prompt_index,
        read.csv(filename, header = FALSE)
      )
    )
  }
}

df <- data.frame(df)
names(df) <- c(
  "LLM", "GenerationMethod", "PromptIndex", "Metric 1", "Metric 2", "Metric 3"
)
df <- as.data.frame(lapply(df, unlist))

for (model_prefix in models_folders_prefix) {
  hist_df <- rbind(
    cbind("Metric 1", df[df$LLM == model_prefix, "Metric.1"]),
    cbind("Metric 2", df[df$LLM == model_prefix, "Metric.2"]),
    cbind("Metric 3", df[df$LLM == model_prefix, "Metric.3"])
  )
  hist_df <- data.frame(hist_df)
  names(hist_df) <- c("Metric", "Value")
  hist_df <- as.data.frame(lapply(hist_df, unlist))
  hist_df$Metric <-
    factor(hist_df$Metric, levels = c("Metric 1", "Metric 3", "Metric 2"))
  hist_df$Value <- as.numeric(hist_df$Value)

  print(aggregate(hist_df$Value <= 0.05, by = list(hist_df$Metric), FUN = mean))
  print(aggregate(hist_df$Value <= 0.01, by = list(hist_df$Metric), FUN = mean))

  p <- ggplot2::ggplot() +
    ggplot2::geom_histogram(
      ggplot2::aes(x = Value, fill = Metric),
      data = hist_df,
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
}

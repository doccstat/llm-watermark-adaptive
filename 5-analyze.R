set.seed(1)

folder <- "results/"
models <- c("meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.1")
models_folders_prefix <- c("ml3", "mt7")
generation_methods <- c("gumbel")
attacks <- c("deletion", "insertion", "substitution")
watermark_key_token_pairs <- matrix(c(
  10, 10,
  20, 20,
  30, 30
  # 40, 40,
  # 50, 50,
  # 100, 100
), ncol = 2, byrow = TRUE)
attack_pcts <- c(
  "0.0", "0.1", "0.2", "0.3"
)
watermarked_or_null <- c("watermarked")

pvalue_files_templates <- matrix(NA, 0, 15)
for (wkt_index in seq_len(nrow(watermark_key_token_pairs))) {  # nolint
  watermark_key_length <- watermark_key_token_pairs[wkt_index, 1]
  tokens_count <- watermark_key_token_pairs[wkt_index, 2]
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
              watermark_key_length,
              "-",
              tokens_count,
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
}

prompt_count <- 100
dfs <- list()
filename <- sub("XXX", 0, paste0(pvalue_files_templates[1, ], collapse = ""))
metric_count <- ncol(read.csv(filename, header = FALSE))

clusters <- parallel::makeCluster(parallel::detectCores() - 1)
doParallel::registerDoParallel(clusters)
for (template_index in seq_len(nrow(pvalue_files_templates))) {
  print(paste("Processing", template_index, "of", nrow(pvalue_files_templates)))
  probs <- tryCatch(
    read.csv(
      paste0(
        paste0(
          pvalue_files_templates[template_index, seq_len(12)], collapse = ""
        ), ".p-probs.csv"
      ), header = FALSE
    ),
    error = function(e) {
      matrix(
        NA, prompt_count, as.numeric(pvalue_files_templates[template_index, 10])
      )
    }
  )
  empty_probs <- tryCatch(
    read.csv(
      paste0(
        paste0(
          pvalue_files_templates[template_index, seq_len(12)], collapse = ""
        ), ".p-empty-probs.csv"
      ), header = FALSE
    ),
    error = function(e) {
      matrix(
        NA, prompt_count, as.numeric(pvalue_files_templates[template_index, 10])
      )
    }
  )
  pvalues_matrix <- foreach::`%dopar%`(
    foreach::foreach(prompt_index = seq_len(prompt_count), .combine = "cbind"),
    {
      filename <- sub(
        "XXX",
        prompt_index - 1,
        paste0(pvalue_files_templates[template_index, ], collapse = "")
      )
      matrix(tryCatch(
        read.csv(filename, header = FALSE),
        error = function(e) rep(NA, metric_count)
      ))[seq_len(metric_count), ]
    }
  )
  for (prompt_index in seq_len(prompt_count)) {
    # Python index in the file name
    filename <- sub(
      "XXX",
      prompt_index - 1,
      paste0(pvalue_files_templates[template_index, ], collapse = "")
    )
    dfs[[prompt_count * (template_index - 1) + prompt_index]] <- cbind(
      pvalue_files_templates[template_index, 2],
      pvalue_files_templates[template_index, 4],
      pvalue_files_templates[template_index, 6],
      pvalue_files_templates[template_index, 8],
      pvalue_files_templates[template_index, 10],
      pvalue_files_templates[template_index, 12],
      prompt_index,
      seq_len(metric_count),
      pvalues_matrix[, prompt_index],
      sum(abs(probs[prompt_index, ] - empty_probs[prompt_index, ])),
      sqrt(sum((probs[prompt_index, ] - empty_probs[prompt_index, ])^2)),
      max(abs(probs[prompt_index, ] - empty_probs[prompt_index, ]))
    )
  }
}
parallel::stopCluster(clusters)

df <- do.call(rbind, dfs)
df <- data.frame(df)
names(df) <- c(
  "LLM", "GenerationMethod", "Attack", "WatermarkKeyLength", "TokensCount",
  "AttackPct", "PromptIndex", "Metric", "PValue",
  "ProbsErrorL1Norm", "ProbsErrorL2Norm", "ProbsErrorInfNorm"
)
df <- as.data.frame(lapply(df, unlist))  # nolint
df$LLM <- as.character(df$LLM)
df$AttackPct <- as.numeric(df$AttackPct)
df$WatermarkKeyLength <- as.numeric(df$WatermarkKeyLength)
df$TokensCount <- as.numeric(df$TokensCount)
df$PromptIndex <- as.numeric(df$PromptIndex)
df$Metric <- as.factor(df$Metric)
df$PValue <- as.numeric(df$PValue)
df$ProbsErrorL1Norm <- as.numeric(df$ProbsErrorL1Norm)
df$ProbsErrorL2Norm <- as.numeric(df$ProbsErrorL2Norm)
df$ProbsErrorInfNorm <- as.numeric(df$ProbsErrorInfNorm)

################################################################################

correct_identified <- matrix(NA, nrow(pvalue_files_templates), 7)
for (template_index in seq_len(nrow(pvalue_files_templates))) {
  best_prompt <- as.matrix(read.csv(
    paste0(
      paste0(
        pvalue_files_templates[template_index, seq_len(12)], collapse = ""
      ), ".p-best-prompt.csv"
    ),
    header = FALSE
  ))
  true_prompt <- as.matrix(read.csv(
    paste0(
      paste0(
        pvalue_files_templates[template_index, seq_len(12)], collapse = ""
      ), ".p-prompt.csv"
    ),
    header = FALSE
  ))
  correct_identified[template_index, ] <- c(
    pvalue_files_templates[template_index, c(2, 4, 6, 8, 10, 12)],
    sum(rowSums(best_prompt == true_prompt) >= 45)
  )
}
correct_identified <- as.data.frame(correct_identified)
names(correct_identified) <- c(
  "LLM", "GenerationMethod", "Attack", "WatermarkKeyLength", "TokensCount",
  "AttackPct", "CorrectIdentified"
)
correct_identified$LLM <- as.character(correct_identified$LLM)
correct_identified$GenerationMethod <- as.character(correct_identified$GenerationMethod)
correct_identified$Attack <- as.character(correct_identified$Attack)
correct_identified$WatermarkKeyLength <- as.numeric(correct_identified$WatermarkKeyLength)
correct_identified$TokensCount <- factor(correct_identified$TokensCount)
correct_identified$AttackPct <- as.numeric(correct_identified$AttackPct)
correct_identified$CorrectIdentified <- as.numeric(correct_identified$CorrectIdentified) / 100

for (model_prefix in models_folders_prefix) {
  p <- ggplot2::ggplot(
    correct_identified[correct_identified$LLM == model_prefix, ],
    ggplot2::aes(x = AttackPct, y = CorrectIdentified, color = TokensCount)
  ) +
    ggplot2::geom_line() +
    ggplot2::facet_grid(
      ~ Attack,
      labeller = ggplot2::label_both
    ) +
    ggplot2::xlab("Attack percentage") +
    ggplot2::ylab("Correctly identified") +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "bottom")
  ggplot2::ggsave(
    paste0("results/", model_prefix, "-correct-identified.pdf"), p, width = 7, height = 3
  )
}

################################################################################

interested_metrics <- c(1, 2, 11, 24)

theoretical_df <- data.frame(df[
  df$GenerationMethod == "gumbel" &
    df$Attack == "deletion" &
    df$AttackPct == 0,
  c(
    "LLM", "WatermarkKeyLength", "TokensCount",
    "PromptIndex", "Metric", "PValue"
  )
])
theoretical_df_power <- aggregate(
  theoretical_df$PValue <= 0.05,
  by = list(
    LLM = theoretical_df$LLM,
    WatermarkKeyLength = theoretical_df$WatermarkKeyLength,
    TokensCount = theoretical_df$TokensCount,
    Metric = theoretical_df$Metric
  ),
  FUN = function(x) mean(x, na.rm = TRUE)
)

theoretical_df_power_i <- theoretical_df_power[
  theoretical_df_power$Metric %in% interested_metrics,
]

color_palette <- seq_along(interested_metrics)
names(color_palette) <- interested_metrics

for (model_prefix in models_folders_prefix) {
  theoretical_df_power_i_llm <- theoretical_df_power_i[
    theoretical_df_power_i$LLM == model_prefix,
  ]
  theoretical_df_power_i_llm$Metric <- factor(
    theoretical_df_power_i_llm$Metric,
    levels = interested_metrics[color_palette]
  )
  p <- ggplot2::ggplot(
    theoretical_df_power_i_llm,
    ggplot2::aes(x = TokensCount, y = x, color = Metric)
  ) +
    ggplot2::geom_line() +
    ggplot2::scale_color_hue(labels = color_palette) +
    ggplot2::theme_minimal() +
    ggplot2::xlab("Text length") +
    ggplot2::ylab("Power") +
    ggplot2::scale_x_continuous(
      breaks = c(10, 20, 30),
      labels = c(10, 20, 30)
    )
  ggplot2::ggsave(
    paste0("results/", model_prefix, "-theoretical.pdf"), p, width = 4.2, height = 1.6
  )
}

################################################################################
################################################################################

probs_true_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-probs.csv"
probs_98_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-98-probs.csv"
probs_96_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-96-probs.csv"
probs_90_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-90-probs.csv"
probs_80_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-80-probs.csv"
probs_60_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-60-probs.csv"
probs_40_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-40-probs.csv"
probs_20_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-20-probs.csv"
probs_empty_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-empty-probs.csv"

probs_true <- as.matrix(read.csv(probs_true_filename, header = FALSE))[c(1, 3, 30), ]
probs_98 <- as.matrix(read.csv(probs_98_filename, header = FALSE))[c(1, 3, 30), ]
probs_96 <- as.matrix(read.csv(probs_96_filename, header = FALSE))[c(1, 3, 30), ]
probs_90 <- as.matrix(read.csv(probs_90_filename, header = FALSE))[c(1, 3, 30), ]
probs_80 <- as.matrix(read.csv(probs_80_filename, header = FALSE))[c(1, 3, 30), ]
probs_60 <- as.matrix(read.csv(probs_60_filename, header = FALSE))[c(1, 3, 30), ]
probs_40 <- as.matrix(read.csv(probs_40_filename, header = FALSE))[c(1, 3, 30), ]
probs_20 <- as.matrix(read.csv(probs_20_filename, header = FALSE))[c(1, 3, 30), ]
probs_empty <- as.matrix(read.csv(probs_empty_filename, header = FALSE))[c(1, 3, 30), ]

df_probs <- data.frame(
  rbind(
    cbind(0, seq_len(nrow(probs_true)), probs_true),
    cbind(1, seq_len(nrow(probs_98)), probs_98),
    cbind(2, seq_len(nrow(probs_96)), probs_96),
    cbind(5, seq_len(nrow(probs_90)), probs_90),
    cbind(10, seq_len(nrow(probs_80)), probs_80),
    cbind(20, seq_len(nrow(probs_60)), probs_60),
    cbind(30, seq_len(nrow(probs_40)), probs_40),
    cbind(40, seq_len(nrow(probs_20)), probs_20),
    cbind(50, seq_len(nrow(probs_empty)), probs_empty)
  )
)
names(df_probs) <- c("EditDistance", "PromptIndex", paste0("Probs", seq_len(ncol(probs_true))))
interested_edit_distances <- c(0, 10, 30, 50)
df_probs_molten <- reshape2::melt(
  df_probs[df_probs$EditDistance %in% interested_edit_distances, ],
  id.vars = c("EditDistance", "PromptIndex"),
  variable.name = "TokenIndex",
  value.name = "Probs"
)
df_probs_molten$EditDistance <- factor(
  df_probs_molten$EditDistance,
  levels = interested_edit_distances
)
df_probs_molten$TokenIndex <- as.numeric(df_probs_molten$TokenIndex)

p <- ggplot2::ggplot(
  df_probs_molten,
  ggplot2::aes(
    x = TokenIndex, y = Probs, color = EditDistance, group = EditDistance
  )
) +
  ggplot2::geom_line() +
  ggplot2::scale_y_continuous(limits = c(0, 1)) +
  ggplot2::facet_wrap(~ PromptIndex, nrow = 1) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    legend.position = "bottom", strip.text.x = ggplot2::element_blank()
  ) +
  ggplot2::xlab("Token index") +
  ggplot2::ylab("Probability") +
  ggplot2::scale_color_hue(
    labels = c("0", "10", "30", "50"), name = "Edit distance"
  ) +
  ggplot2::scale_x_continuous(breaks = c(1, 10, 20)) +
  ggplot2::guides(color = ggplot2::guide_legend(nrow = 1, byrow = TRUE))
ggplot2::ggsave("results/probs-edit-distance.pdf", p, width = 10, height = 2.2)

################################################################################
################################################################################

# The following piece code is no longer used.
probs_true_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-probs.csv"
probs_98_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-98-probs.csv"
probs_96_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-96-probs.csv"
probs_90_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-90-probs.csv"
probs_80_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-80-probs.csv"
probs_60_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-60-probs.csv"
probs_40_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-40-probs.csv"
probs_20_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-20-probs.csv"
probs_empty_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-empty-probs.csv"

probs_true <- as.matrix(read.csv(probs_true_filename, header = FALSE))[seq_len(100), ]
probs_98 <- as.matrix(read.csv(probs_98_filename, header = FALSE))[seq_len(100), ]
probs_96 <- as.matrix(read.csv(probs_96_filename, header = FALSE))[seq_len(100), ]
probs_90 <- as.matrix(read.csv(probs_90_filename, header = FALSE))[seq_len(100), ]
probs_80 <- as.matrix(read.csv(probs_80_filename, header = FALSE))[seq_len(100), ]
probs_60 <- as.matrix(read.csv(probs_60_filename, header = FALSE))[seq_len(100), ]
probs_40 <- as.matrix(read.csv(probs_40_filename, header = FALSE))[seq_len(100), ]
probs_20 <- as.matrix(read.csv(probs_20_filename, header = FALSE))[seq_len(100), ]
probs_empty <- as.matrix(read.csv(probs_empty_filename, header = FALSE))[seq_len(100), ]

df_probs <- data.frame(
  rbind(
    cbind(0, seq_len(nrow(probs_true)), probs_true),
    cbind(1, seq_len(nrow(probs_98)), probs_98),
    cbind(2, seq_len(nrow(probs_96)), probs_96),
    cbind(5, seq_len(nrow(probs_90)), probs_90),
    cbind(10, seq_len(nrow(probs_80)), probs_80),
    cbind(20, seq_len(nrow(probs_60)), probs_60),
    cbind(30, seq_len(nrow(probs_40)), probs_40),
    cbind(40, seq_len(nrow(probs_20)), probs_20),
    cbind(50, seq_len(nrow(probs_empty)), probs_empty)
  )
)
names(df_probs) <- c("EditDistance", "PromptIndex", paste0("Probs", seq_len(ncol(probs_true))))
interested_edit_distances <- c(0, 1, 2, 5, 10, 20, 30, 40, 50)
df_probs$EditDistance <- factor(df_probs$EditDistance, levels = interested_edit_distances)
for (i in seq_len(nrow(df_probs))) {
  df_probs[i, "DTW"] <- dtw::dtw(
    df_probs[i, 2 + seq_len(10)], df_probs[(i - 1) %% 100 + 1, 2 + seq_len(10)]
  )$distance
}

ggplot2::ggplot(
  df_probs,
  ggplot2::aes(x = EditDistance, y = DTW, color = EditDistance)
) +
  ggplot2::geom_boxplot() +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "bottom")

################################################################################
################################################################################

# The following piece code is no longer used.
probs_true_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-probs.csv"
probs_98_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-98-probs.csv"
probs_96_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-96-probs.csv"
probs_90_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-90-probs.csv"
probs_80_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-80-probs.csv"
probs_60_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-60-probs.csv"
probs_40_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-40-probs.csv"
probs_20_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-20-probs.csv"
probs_empty_filename <- "results/ml3-gumbel-deletion-10-10-0.0.p-re-calculated-empty-probs.csv"

probs_true <- as.matrix(read.csv(probs_true_filename, header = FALSE))[seq_len(100), ]
probs_98 <- as.matrix(read.csv(probs_98_filename, header = FALSE))[seq_len(100), ]
probs_96 <- as.matrix(read.csv(probs_96_filename, header = FALSE))[seq_len(100), ]
probs_90 <- as.matrix(read.csv(probs_90_filename, header = FALSE))[seq_len(100), ]
probs_80 <- as.matrix(read.csv(probs_80_filename, header = FALSE))[seq_len(100), ]
probs_60 <- as.matrix(read.csv(probs_60_filename, header = FALSE))[seq_len(100), ]
probs_40 <- as.matrix(read.csv(probs_40_filename, header = FALSE))[seq_len(100), ]
probs_20 <- as.matrix(read.csv(probs_20_filename, header = FALSE))[seq_len(100), ]
probs_empty <- as.matrix(read.csv(probs_empty_filename, header = FALSE))[seq_len(100), ]

df_probs <- data.frame(
  rbind(
    cbind(0, seq_len(nrow(probs_true)), probs_true),
    cbind(1, seq_len(nrow(probs_98)), probs_98),
    cbind(2, seq_len(nrow(probs_96)), probs_96),
    cbind(5, seq_len(nrow(probs_90)), probs_90),
    cbind(10, seq_len(nrow(probs_80)), probs_80),
    cbind(20, seq_len(nrow(probs_60)), probs_60),
    cbind(30, seq_len(nrow(probs_40)), probs_40),
    cbind(40, seq_len(nrow(probs_20)), probs_20),
    cbind(50, seq_len(nrow(probs_empty)), probs_empty)
  )
)
names(df_probs) <- c("EditDistance", "PromptIndex", paste0("Probs", seq_len(ncol(probs_true))))
interested_edit_distances <- c(0, 1, 2, 5, 10, 20, 30, 40, 50)
df_probs$EditDistance <- factor(df_probs$EditDistance, levels = interested_edit_distances)
df_probs$criteria <- apply(df_probs[, 2 + seq_len(10)], 1, function(x) sum(log(x)))

ggplot2::ggplot(
  df_probs[df_probs$EditDistance %in% interested_edit_distances, ],
  ggplot2::aes(x = EditDistance, y = criteria, color = EditDistance)
) +
  ggplot2::geom_boxplot() +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "bottom")

probs_true_index <- apply(probs_true, 1, function(x) which.min(x))
df_probs$TrueMinIndex <- rep(probs_true_index, length(interested_edit_distances))
df_probs$MinIndex <- apply(df_probs[, 2 + seq_len(10)], 1, function(x) which.min(x))
df_probs$CorrectMinIndex <- df_probs$TrueMinIndex == df_probs$MinIndex
df_probs_molten <- reshape2::melt(
  df_probs[df_probs$EditDistance %in% interested_edit_distances, c(
    "EditDistance", "CorrectMinIndex"
  )],
  id.vars = c("EditDistance"),
  value.name = "CorrectMinIndex"
)
df_probs_molten$EditDistance <- factor(
  df_probs_molten$EditDistance,
  levels = interested_edit_distances
)
aggregate(df_probs_molten$CorrectMinIndex, by = list(df_probs_molten$EditDistance), FUN = sum)

################################################################################
################################################################################

# The following piece code is no longer used.
probs_true_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-probs.csv"
probs_98_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-98-probs.csv"
probs_96_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-96-probs.csv"
probs_90_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-90-probs.csv"
probs_80_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-80-probs.csv"
probs_60_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-60-probs.csv"
probs_40_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-40-probs.csv"
probs_20_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-20-probs.csv"
probs_empty_filename <- "results/ml3-gumbel-deletion-20-20-0.0.p-re-calculated-empty-probs.csv"

probs_true <- as.matrix(read.csv(probs_true_filename, header = FALSE))[seq_len(100), ]
probs_98 <- as.matrix(read.csv(probs_98_filename, header = FALSE))[seq_len(100), ]
probs_96 <- as.matrix(read.csv(probs_96_filename, header = FALSE))[seq_len(100), ]
probs_90 <- as.matrix(read.csv(probs_90_filename, header = FALSE))[seq_len(100), ]
probs_80 <- as.matrix(read.csv(probs_80_filename, header = FALSE))[seq_len(100), ]
probs_60 <- as.matrix(read.csv(probs_60_filename, header = FALSE))[seq_len(100), ]
probs_40 <- as.matrix(read.csv(probs_40_filename, header = FALSE))[seq_len(100), ]
probs_20 <- as.matrix(read.csv(probs_20_filename, header = FALSE))[seq_len(100), ]
probs_empty <- as.matrix(read.csv(probs_empty_filename, header = FALSE))[seq_len(100), ]

probs_diff <- data.frame(
  rbind(
    cbind(1, seq_len(nrow(probs_98)), probs_true - probs_98),
    cbind(2, seq_len(nrow(probs_96)), probs_true - probs_96),
    cbind(5, seq_len(nrow(probs_90)), probs_true - probs_90),
    cbind(10, seq_len(nrow(probs_80)), probs_true - probs_80),
    cbind(20, seq_len(nrow(probs_60)), probs_true - probs_60),
    cbind(30, seq_len(nrow(probs_40)), probs_true - probs_40),
    cbind(40, seq_len(nrow(probs_20)), probs_true - probs_20),
    cbind(50, seq_len(nrow(probs_empty)), probs_true - probs_empty)
  )
)
names(probs_diff) <- c(
  "EditDistance", "PromptIndex", paste0("Probs", seq_len(ncol(probs_true)))
)
interested_edit_distances <- c(1, 2, 5, 10, 20, 30, 40, 50)
probs_diff$L1 <- apply(probs_diff[, 2 + seq_len(20)], 1, function(x) sum(abs(x)))
probs_diff$L2 <- apply(probs_diff[, 2 + seq_len(20)], 1, function(x) sqrt(sum(x^2)))
probs_diff$infty <- apply(probs_diff[, 2 + seq_len(20)], 1, function(x) max(abs(x)))
probs_true_index <- apply(probs_true, 1, function(x) which.min(x))
probs_diff$MinProbError <- abs(probs_diff[, 2 + seq_len(20)][cbind(seq_len(nrow(probs_diff)), rep(probs_true_index, length(interested_edit_distances)))])
probs_diff_molten <- reshape2::melt(
  probs_diff[probs_diff$EditDistance %in% interested_edit_distances, c(
    "EditDistance", "L1", "L2", "infty", "MinProbError"
  )],
  id.vars = c("EditDistance"),
  value.name = "value"
)
probs_diff_molten$EditDistance <- factor(
  probs_diff_molten$EditDistance,
  levels = interested_edit_distances
)

p <- ggplot2::ggplot(
  probs_diff_molten,
  ggplot2::aes(x = EditDistance, y = value, fill = EditDistance)
) +
  ggplot2::geom_boxplot() +
  ggplot2::scale_y_continuous(trans = "log10") +
  ggplot2::facet_wrap(~ variable, scales = "free") +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "bottom")
ggplot2::ggsave("results/probs-diff.pdf", p, width = 5, height = 3)

################################################################################
################################################################################

# The following piece code is no longer used.
theoretical_df_power_i$label_x <- prompt_count + 3
for (llm in unique(theoretical_df_power_i$LLM)) {  # nolint
  for (wkl in unique(theoretical_df_power_i$WatermarkKeyLength)) {
    facet_df <- theoretical_df_power_i[
      theoretical_df_power_i$LLM == llm &
        theoretical_df_power_i$WatermarkKeyLength == wkl,
    ]
    facet_df <- facet_df[order(facet_df$x, decreasing = TRUE), ]
    y_i_offset <- 0
    last_first <- 1
    for (y_i in seq_len(nrow(facet_df))) {
      if (
        y_i > 1 &&
          !is.na(facet_df[last_first, "x"]) &&
          !is.na(facet_df[y_i, "x"]) &&
          facet_df[last_first, "x"] - facet_df[y_i, "x"] > 0.05
      ) {
        y_i_offset <- y_i - 1
        last_first <- y_i
      }
      theoretical_df_power_i[
        theoretical_df_power_i$LLM == llm &
          theoretical_df_power_i$WatermarkKeyLength == wkl &
          theoretical_df_power_i$Metric == facet_df[y_i, "Metric"],
        "label_x"
      ] <- prompt_count + 5 + 12 * ((y_i - y_i_offset - 1) %% 7)
    }
  }
}
theoretical_df_power_ni <- theoretical_df_power[
  !(theoretical_df_power$Metric %in% interested_metrics),
]

p <- ggplot2::ggplot(
  theoretical_df,
  ggplot2::aes(x = PromptIndex, y = PValue, color = Metric)
) +
  ggplot2::geom_point(alpha = 0.02) +
  ggplot2::geom_segment(
    data = theoretical_df_power_i,
    ggplot2::aes(
      x = 0,
      xend = prompt_count,
      y = x,
      yend = x,
      color = Metric
    )
  ) +
  ggplot2::geom_segment(
    data = theoretical_df_power_ni,
    ggplot2::aes(
      x = 0,
      xend = prompt_count,
      y = x,
      yend = x,
      color = Metric
    ),
    alpha = 0.2
  ) +
  ggplot2::geom_text(
    data = theoretical_df_power_i,
    ggplot2::aes(label = Metric, x = label_x, y = x),
    size = 2.8
  ) +
  ggplot2::annotate(
    "segment",
    x = 0,
    xend = prompt_count,
    y = 0.05,
    yend = 0.05,
    color = "black",
    linetype = "dashed",
    alpha = 0.5
  ) +
  ggplot2::scale_x_continuous(
    breaks = c(0, 50, 100),
    labels = c(0, 50, 100)
  ) +
  ggplot2::facet_grid(LLM ~ WatermarkKeyLength) +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "none")
ggplot2::ggsave("results/theoretical.pdf", p, width = 7, height = 5)

################################################################################
################################################################################

powers <- rbind(
  cbind(
    Threshold = 0.05,
    aggregate(
      df$PValue <= 0.05,
      by = list(
        LLM = df$LLM,
        GenerationMethod = df$GenerationMethod,
        Attack = df$Attack,
        WatermarkKeyLength = df$WatermarkKeyLength,
        TokensCount = df$TokensCount,
        AttackPct = df$AttackPct,
        Metric = df$Metric
      ),
      FUN = function(x) c(Mean = mean(x, na.rm = TRUE), StdDev = sd(x, na.rm = TRUE))
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
        WatermarkKeyLength = df$WatermarkKeyLength,
        TokensCount = df$TokensCount,
        AttackPct = df$AttackPct,
        Metric = df$Metric
      ),
      FUN = function(x) c(Mean = mean(x, na.rm = TRUE), StdDev = sd(x, na.rm = TRUE))
    )
  )
)
powers <- do.call(data.frame, powers)
powers$Metric <- factor(powers$Metric, levels = seq_len(metric_count))
powers <- powers[order(
  powers$WatermarkKeyLength,
  powers$TokensCount,
  powers$Threshold,
  powers$LLM,
  powers$GenerationMethod,
  powers$Metric
), ]

for (model_prefix in models_folders_prefix) {
  for (threshold in c(0.05)) {
    for (attack in attacks) {
      print(paste(model_prefix, threshold, attack, "power"))
      tab <- NULL
      for (tct in watermark_key_token_pairs[, 2]) {
        tab <- cbind(
          tab,
          matrix(
            powers[powers$LLM == model_prefix &
              powers$Threshold == threshold &
              powers$Attack == attack &
              powers$TokensCount == tct &
              powers$Metric %in% interested_metrics, "x.Mean"],
            ncol = length(attack_pcts),
            byrow = TRUE
          )
        )
      }
      rownames(tab) <- paste("Metric", color_palette)
      tab <- cbind(tab, rowSums(tab[, -c(1, 5, 9)]))
      print(xtable::xtable(tab, type = "latex", digits = 2))

      # print(paste(model_prefix, threshold, attack, "stddev"))
      # tab <- NULL
      # for (tct in watermark_key_token_pairs[, 2]) {
      #   tab <- cbind(
      #     tab,
      #     matrix(
      #       powers[powers$LLM == model_prefix &
      #         powers$Threshold == threshold &
      #         powers$Attack == attack &
      #         powers$TokensCount == tct &
      #         powers$Metric %in% interested_metrics, "x.StdDev"],
      #       ncol = length(attack_pcts),
      #       byrow = TRUE
      #     )
      #   )
      # }
      # rownames(tab) <- paste("Metric", color_palette)
      # print(xtable::xtable(tab, type = "latex", digits = 2))
    }
  }
}

# powers$LineType <- rep("dashed", nrow(powers))
# powers$LineType[powers$Metric == 2] <- "theoretical"
# powers$LineType[powers$Metric %in% (2 + seq_len(13))] <- "empty"
# powers$LineType[powers$Metric %in% (15 + seq_len(13))] <- "best"
# powers$LineType[powers$Metric %in% (28 + seq_len(13))] <- "icl"

# powers$line_alpha <- 0.2
# powers$line_alpha[powers$Metric %in% interested_metrics] <- 1

# matrix(powers[
#   powers$Attack == "substitution" &
#     powers$GenerationMethod == "gumbel" &
#     powers$Threshold == 0.05 &
#     powers$LLM == "ml3" &
#     powers$WatermarkKeyLength == 100 &
#     powers$TokensCount == 100,
#   "x"
# ], nrow = metric_count, byrow = TRUE)

# color_palette <- c(
#   "black", grDevices::hcl(
#     h = seq(15, 375, length = length(interested_metrics) + 1), l = 65, c = 100
#   )[seq_along(interested_metrics)]
# )
# names(color_palette) <- c("black", as.character(interested_metrics))

for (threshold in c(0.05, 0.01)) {
  for (model_prefix in models_folders_prefix) {
    p <- ggplot2::ggplot() +
      ggplot2::geom_line(
        ggplot2::aes(
          x = AttackPct,
          y = x.Mean,
          color = Metric
        ),
        data = powers[
          powers$Threshold == threshold &
            powers$LLM == model_prefix &
            powers$Metric %in% interested_metrics,
        ]
      ) +
      ggplot2::facet_grid(
        Attack ~ TokensCount, scales = "free_y"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::scale_x_continuous(labels = scales::percent) +
      ggplot2::scale_color_hue(labels = color_palette) +
      ggplot2::theme(legend.position = "bottom") +
      ggplot2::xlab("Attack percentage") +
      ggplot2::ylab("Power") +
      ggplot2::guides(
        linetype = "none",
        alpha = "none",
        color = ggplot2::guide_legend(nrow = 1, byrow = TRUE)
      )
    ggplot2::ggsave(
      paste0(
        "results/",
        model_prefix,
        "-powers-",
        threshold,
        ".pdf"
      ),
      p,
      width = 10,
      height = 5
    )
  }
}

################################################################################
################################################################################

# The following piece code is no longer used.
watermark_key_length <- 10
tokens_count <- 10
llm <- "ml3"

df_probs_list <- list()

probs_matrix <- read.csv(
  paste0(
    folder,
    llm,
    "-gumbel-deletion-",
    watermark_key_length,
    "-",
    tokens_count,
    "-0.0.p-probs.csv"
  ),
  header = FALSE
)
probs_name <- "true"
df_probs_list[[probs_name]] <- cbind(
  probs_name,
  seq_len(nrow(probs_matrix)),
  probs_matrix
)

probs_matrix <- read.csv(
  paste0(
    folder,
    llm,
    "-gumbel-deletion-",
    watermark_key_length,
    "-",
    tokens_count,
    "-0.0.p-re-calculated-empty-probs.csv"
  ),
  header = FALSE
)
probs_name <- "empty"
df_probs_list[[probs_name]] <- cbind(
  probs_name,
  seq_len(nrow(probs_matrix)),
  probs_matrix
)

probs_matrix <- read.csv(
  paste0(
    folder,
    llm,
    "-gumbel-deletion-",
    watermark_key_length,
    "-",
    tokens_count,
    "-0.0.p-re-calculated-best-probs.csv"
  ),
  header = FALSE
)
probs_name <- "best"
df_probs_list[[probs_name]] <- cbind(
  probs_name,
  seq_len(nrow(probs_matrix)),
  probs_matrix
)

probs_matrix <- read.csv(
  paste0(
    folder,
    llm,
    "-gumbel-deletion-",
    watermark_key_length,
    "-",
    tokens_count,
    "-0.0.p-re-calculated-icl-probs.csv"
  ),
  header = FALSE
)
probs_name <- "icl"
df_probs_list[[probs_name]] <- cbind(
  probs_name,
  seq_len(nrow(probs_matrix)),
  probs_matrix
)

difference_df <- list()
for (probs_name in c("empty", "best", "icl")) {
  difference_df[[probs_name]] <- cbind(
    probs_name,
    seq_len(nrow(df_probs_list[[probs_name]])),
    df_probs_list[["true"]][, -c(1:2)] - df_probs_list[[probs_name]][, -c(1:2)]
  )
  names(difference_df[[probs_name]]) <-
    c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))
}
for (lambda in 0:9 / 10) {
  probs_name <- paste0("empty,S1-", 1 - lambda)
  difference_df[[probs_name]] <- cbind(
    probs_name,
    seq_len(nrow(df_probs_list[["empty"]])),
    df_probs_list[["true"]][, -c(1, 2)] - (
      lambda * df_probs_list[["empty"]][, -c(1, 2)] + (1 - lambda) * 0.5
    )
  )
  names(difference_df[[probs_name]]) <-
    c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))
}

exp1 <- df_probs_list[["empty"]][, -c(1, 2)]
exp1[exp1 >= 0.1] <- min(1, 3 * exp1[exp1 >= 0.1])
difference_df[["exp1"]] <- cbind(
  "exp1",
  seq_len(nrow(df_probs_list[["empty"]])),
  df_probs_list[["true"]][, -c(1, 2)] - exp1
)
names(difference_df[["exp1"]]) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))

exp2 <- df_probs_list[["empty"]][, -c(1, 2)]
exp2[exp2 >= 0.1] <- min(1, 2 * exp2[exp2 >= 0.1])
difference_df[["exp2"]] <- cbind(
  "exp2",
  seq_len(nrow(df_probs_list[["empty"]])),
  df_probs_list[["true"]][, -c(1, 2)] - exp2
)
names(difference_df[["exp2"]]) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))

exp3 <- df_probs_list[["empty"]][, -c(1, 2)]
exp3 <- 0.8 * exp3 + 0.2
difference_df[["exp3"]] <- cbind(
  "exp3",
  seq_len(nrow(df_probs_list[["empty"]])),
  df_probs_list[["true"]][, -c(1, 2)] - exp3
)
names(difference_df[["exp3"]]) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))

exp4 <- df_probs_list[["empty"]][, -c(1, 2)]
exp4[exp4 >= 0.1] <- 0.8 * exp4[exp4 >= 0.1] + 0.2
difference_df[["exp4"]] <- cbind(
  "exp4",
  seq_len(nrow(df_probs_list[["empty"]])),
  df_probs_list[["true"]][, -c(1, 2)] - exp4
)
names(difference_df[["exp4"]]) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))

exp5 <- df_probs_list[["empty"]][, -c(1, 2)]
exp5[, 1] <- 0.5
difference_df[["exp5"]] <- cbind(
  "exp5",
  seq_len(nrow(df_probs_list[["empty"]])),
  df_probs_list[["true"]][, -c(1, 2)] - exp5
)
names(difference_df[["exp5"]]) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))

exp6 <- df_probs_list[["empty"]][, -c(1, 2)]
exp6[exp6 <= 1e-6] <- 0.9
difference_df[["exp6"]] <- cbind(
  "exp6",
  seq_len(nrow(df_probs_list[["empty"]])),
  df_probs_list[["true"]][, -c(1, 2)] - exp6
)
names(difference_df[["exp6"]]) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))

exp7 <- df_probs_list[["empty"]][, -c(1, 2)]
exp7 <- sqrt(exp7)
difference_df[["exp7"]] <- cbind(
  "exp7",
  seq_len(nrow(df_probs_list[["empty"]])),
  df_probs_list[["true"]][, -c(1, 2)] - exp7
)
names(difference_df[["exp7"]]) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))

exp8 <- df_probs_list[["empty"]][, -c(1, 2)]
exp8 <- exp8^(1/3)
difference_df[["exp8"]] <- cbind(
  "exp8",
  seq_len(nrow(df_probs_list[["empty"]])),
  df_probs_list[["true"]][, -c(1, 2)] - exp8
)
names(difference_df[["exp8"]]) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))

exp9 <- df_probs_list[["empty"]][, -c(1, 2)]
exp9 <- pbeta(as.matrix(exp9), 0.5, 1)
difference_df[["exp9"]] <- cbind(
  "exp9",
  seq_len(nrow(df_probs_list[["empty"]])),
  df_probs_list[["true"]][, -c(1, 2)] - exp9
)
names(difference_df[["exp9"]]) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))

difference_df <- do.call(rbind, difference_df)
difference_df <- data.frame(difference_df)
names(difference_df) <-
  c("ProbsName", "PromptIndex", paste0("Probs", seq_len(tokens_count)))
difference_df$infty <- apply(
  difference_df[, -c(1, 2)],
  1,
  function(x) max(abs(x))
)
difference_df$L2 <- apply(
  difference_df[, -c(1, 2)],
  1,
  function(x) sqrt(sum(x^2))
)
difference_df$L1 <- apply(
  difference_df[, -c(1, 2)],
  1,
  function(x) sum(abs(x))
)
difference_df_molten <- reshape2::melt(
  difference_df[, c("ProbsName", "L1", "L2", "infty")],
  id.vars = c("ProbsName"),
  value.name = "value"
)

p <- ggplot2::ggplot(
  difference_df_molten,
  ggplot2::aes(x = variable, y = value, fill = ProbsName)
) +
  ggplot2::geom_boxplot() +
  ggplot2::facet_wrap(~variable, scales = "free") +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "bottom")
ggplot2::ggsave("results/probs-error-boxplot.pdf", p, width = 15, height = 8)

df_probs <- do.call(rbind, df_probs_list)
df_probs <- data.frame(df_probs)
names(df_probs) <-
  c("ProbsName", "PromptIndex", seq_len(tokens_count))

df_probs_molten <- reshape2::melt(
  df_probs,
  id.vars = c("ProbsName", "PromptIndex"),
  variable.name = "TokenIndex",
  value.name = "Probs"
)
df_probs_molten$TokenIndex <- as.numeric(df_probs_molten$TokenIndex)

p <- ggplot2::ggplot(
  df_probs_molten[df_probs_molten$ProbsName %in% c("true", "empty"), ],
  ggplot2::aes(x = TokenIndex, y = Probs, color = ProbsName, group = ProbsName)
) +
  ggplot2::geom_line() +
  ggplot2::scale_y_continuous(limits = c(0, 0.1)) +
  ggplot2::facet_wrap(~PromptIndex) +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "bottom") +
  ggplot2::scale_x_continuous(breaks = seq_len(tokens_count))
ggplot2::ggsave("results/probs.pdf", p, width = 30, height = 15)

df_probs$criteria <- apply(df_probs[, -c(1, 2)], 1, function(x) sum(log(x)))
p <- ggplot2::ggplot(
  df_probs,
  ggplot2::aes(x = ProbsName, y = criteria, fill = ProbsName)
) +
  ggplot2::geom_boxplot() +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "bottom")
ggplot2::ggsave("results/probs-criteria-boxplot.pdf", p, width = 8, height = 4)

################################################################################
################################################################################

# The following piece code is no longer used.
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

################################################################################
################################################################################

probs_matrix_filename <- "results/ml3-gumbel-substitution-10-10-0.1.p-re-calculated-probs.csv"
attacked_indices_filename <- "results/ml3-gumbel-substitution-10-10-0.1.p-attacked-idx.csv"

probs_matrix <- as.matrix(read.csv(probs_matrix_filename, header = FALSE))
attacked_indices <- as.matrix(read.csv(attacked_indices_filename, header = FALSE))

attacked_probs <- probs_matrix[cbind(seq_len(nrow(probs_matrix)), attacked_indices[, 1] + 1)]
rest_probs <- probs_matrix[-(attacked_indices[, 1] * nrow(probs_matrix) + seq_len(nrow(probs_matrix)))]

p <- ggplot2::ggplot(
    data = data.frame(
        Probs = c(attacked_probs, rest_probs),
        Group = c(rep("Attacked", length(attacked_probs)), rep("Unattacked", length(rest_probs)))
    ),
    ggplot2::aes(x = Group, y = Probs, fill = Group)
) +
    ggplot2::geom_boxplot(width = 0.1, outlier.shape = NA) +
    gghalves::geom_half_point(ggplot2::aes(color = Group), side = "l", range_scale = 0.4, alpha = 0.4, size = 0.5) +
    see::geom_violinhalf(scale = "width", width = 0.8, position = ggplot2::position_nudge(0.1, 0)) +
  ggplot2::coord_flip() +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "none") +
  ggplot2::scale_y_continuous(
    limits = c(0, 1),
    transform = scales::new_transform(
      name = "fifth_root",
      transform = function(y) y^(1/5),
      inverse = function(y) y^5,
      domain = c(0, Inf)
    ),
    breaks = c(1e-9, 0.1, 0.2, 0.6, 1),
    labels = c(1e-9, 0.1, 0.2, 0.6, 1)
  ) +
  ggplot2::xlab("") +
  ggplot2::ylab("Probability")
ggplot2::ggsave("results/probs-attacked-vs-rest.pdf", p, width = 7.5, height = 2.5)

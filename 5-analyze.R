set.seed(1)

folder <- "results/"
models <- c("mistralai/Mistral-7B-v0.1")
models_folders_prefix <- c("mt7")
generation_methods <- c("gumbel")
# attacks <- c("substitution")
attacks <- c("deletion", "insertion", "substitution")
k_tokens_count_ratio_list <- c(1.0)
watermark_key_token_pairs <- matrix(c(
  10, 10,
  20, 20,
  30, 30
), ncol = 2, byrow = TRUE)
attack_pcts <- list(
  "substitution" = c("0.0", "0.1", "0.2", "0.3"),
  # "substitution" = c("1.0"),
  "deletion" = c("1.0"),
  "insertion" = c("1.0")
)

pvalue_files_templates <- matrix(NA, 0, 15)
for (wkt_index in seq_len(nrow(watermark_key_token_pairs))) { # nolint
  watermark_key_length <- watermark_key_token_pairs[wkt_index, 1]
  tokens_count <- watermark_key_token_pairs[wkt_index, 2]
  for (k_tokens_count_ratio in k_tokens_count_ratio_list) {
    max_k <- tokens_count * k_tokens_count_ratio
    for (model_index in seq_along(models)) {
      for (generation_methods_index in seq_along(generation_methods)) {
        for (attack_index in seq_along(attacks)) {
          attack_pcts_seq <- attack_pcts[[attacks[attack_index]]]
          for (attack_pct in attack_pcts_seq) {
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
              attack_pct,
              "-",
              max_k,
              "-detect/watermarked-XXX.csv"
            ))
          }
        }
      }
    }
  }
}

prompt_count <- 1000
dfs <- list()
filename <- sub("XXX", 0, paste0(pvalue_files_templates[1, ], collapse = ""))
metric_count <- ncol(read.csv(filename, header = FALSE))
criteria_count <- nrow(read.csv(filename, header = FALSE))

clusters <- parallel::makeCluster(parallel::detectCores() - 1)
doParallel::registerDoParallel(clusters)
for (template_index in seq_len(nrow(pvalue_files_templates))) {
  print(paste("Processing", template_index, "of", nrow(pvalue_files_templates)))
  pvalues_matrix <- foreach::`%dopar%`(
    foreach::foreach(prompt_index = seq_len(prompt_count), .combine = "cbind"),
    {
      filename <- sub(
        "XXX",
        prompt_index - 1,
        paste0(pvalue_files_templates[template_index, ], collapse = "")
      )
      matrix(t(as.matrix(tryCatch(
        read.csv(filename, header = FALSE),
        error = function(e) stop(paste("Error in", filename, ":", e))
      )))[seq_len(metric_count), seq_len(criteria_count)])
    }
  )
  attacked_indices_filename <- paste0(
    paste0(
      pvalue_files_templates[template_index, seq_len(12)],
      collapse = ""
    ), "-attacked-idx.csv"
  )
  attacked_indices_file <- file(attacked_indices_filename, "r")
  for (prompt_index in seq_len(prompt_count)) {
    # Python index in the file name
    filename <- sub(
      "XXX",
      prompt_index - 1,
      paste0(pvalue_files_templates[template_index, ], collapse = "")
    )
    attacked_idx <- as.numeric(strsplit(readLines(attacked_indices_file, n = 1), ",")[[1]])
    dfs[[prompt_count * (template_index - 1) + prompt_index]] <- cbind(
      pvalue_files_templates[template_index, 2],
      # pvalue_files_templates[template_index, 4],
      pvalue_files_templates[template_index, 6],
      # pvalue_files_templates[template_index, 8],
      pvalue_files_templates[template_index, 10],
      sum(attacked_idx < as.numeric(pvalue_files_templates[template_index, 10])) / as.numeric(pvalue_files_templates[template_index, 10]),
      pvalue_files_templates[template_index, 14],
      prompt_index,
      rep(seq_len(metric_count), times = criteria_count),
      rep(seq_len(criteria_count), each = metric_count),
      pvalues_matrix[, prompt_index]
    )
  }
  close(attacked_indices_file)
}
parallel::stopCluster(clusters)

df <- do.call(rbind, dfs)
df <- data.frame(df)
names(df) <- c(
  "LLM", "Attack", "TokensCount",
  "AttackPct", "B", "PromptIndex", "Metric", "Criteria", "PValue"
)
df <- as.data.frame(lapply(df, unlist)) # nolint
# df$LLM <- as.character(df$LLM)
# df$AttackPct <- as.numeric(df$AttackPct)
# df$B <- as.numeric(df$B)
# df$WatermarkKeyLength <- as.numeric(df$WatermarkKeyLength)
# df$TokensCount <- as.numeric(df$TokensCount)
# df$PromptIndex <- as.numeric(df$PromptIndex)
# df$Metric <- as.factor(df$Metric)
# df$Criteria <- as.factor(df$Criteria)
# df$PValue <- as.numeric(df$PValue)

################################################################################

correct_identified <- matrix(NA, nrow(pvalue_files_templates), 7)
for (template_index in seq_len(nrow(pvalue_files_templates))) {
  best_prompt <- as.matrix(read.csv(
    paste0(
      paste0(
        pvalue_files_templates[template_index, seq_len(12)],
        collapse = ""
      ), "-best-prompt.csv"
    ),
    header = FALSE
  ))
  true_prompt <- as.matrix(read.csv(
    paste0(
      paste0(
        pvalue_files_templates[template_index, seq_len(12)],
        collapse = ""
      ), "-prompt.csv"
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
      ~Attack,
      labeller = ggplot2::label_both
    ) +
    ggplot2::xlab("Attack percentage") +
    ggplot2::ylab("Correctly identified") +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "bottom")
  ggplot2::ggsave(
    paste0("results/", model_prefix, "-correct-identified.pdf"), p,
    width = 7, height = 3
  )
}

################################################################################

interested_metrics <- c(1, 2, 3, 4)
interested_metrics_level <- c(2, 4, 3, 1)
color_palette <- c("baseline", "oracle", "empty", "optim")
names(color_palette) <- interested_metrics
interested_tokens <- c(10, 20, 30)
threshold <- 0.05

theoretical_df <- data.frame(df[
  df$Attack == "substitution" &
    df$AttackPct == 0 &
    df$Metric %in% interested_metrics &
    df$TokensCount %in% interested_tokens,
  c(
    "LLM", "TokensCount", "B", "PromptIndex", "Metric", "Criteria", "PValue"
  )
])
theoretical_df_power <- aggregate(
  theoretical_df$PValue <= threshold,
  by = list(
    LLM = theoretical_df$LLM,
    TokensCount = theoretical_df$TokensCount,
    B = theoretical_df$B,
    Metric = theoretical_df$Metric,
    Criteria = theoretical_df$Criteria
  ),
  FUN = function(x) c(Mean = mean(x, na.rm = TRUE), StdError = sd(x, na.rm = TRUE) / sqrt(length(x)))
)
theoretical_df_power <- do.call(data.frame, theoretical_df_power)

for (model_prefix in models_folders_prefix) {
  theoretical_df_power_llm <- theoretical_df_power[
    theoretical_df_power$LLM == model_prefix,
  ]
  theoretical_df_power_llm$Metric <- factor(
    theoretical_df_power_llm$Metric,
    levels = interested_metrics_level
  )
  theoretical_df_power_llm$B <- as.numeric(theoretical_df_power_llm$B)
  theoretical_df_power_llm$TokensCount <- as.numeric(theoretical_df_power_llm$TokensCount)
  theoretical_df_power_llm$k_tokens_count_ratio <-
    theoretical_df_power_llm$B / theoretical_df_power_llm$TokensCount
  p <- ggplot2::ggplot(
    theoretical_df_power_llm,
    ggplot2::aes(x = TokensCount, y = x.Mean, color = Criteria)
  ) +
    ggplot2::geom_line() +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = x.Mean - x.StdError, ymax = x.Mean + x.StdError),
      width = 0.3
    ) +
    ggplot2::scale_color_hue(labels = color_palette) +
    ggplot2::facet_grid(
       ~ k_tokens_count_ratio,
      labeller = ggplot2::labeller(
        k_tokens_count_ratio = c("0.3" = "B/m: 0.3", "0.6" = "B/m: 0.6", "1" = "B/m: 1.0")
      )
    ) +
    ggplot2::theme_minimal() +
    ggplot2::xlab("Text length") +
    ggplot2::ylab("Power") +
    ggplot2::scale_x_continuous(
      breaks = interested_tokens,
      labels = interested_tokens
    )
  ggplot2::ggsave(
    paste0("results/theoretical-", model_prefix, "-", threshold, ".pdf"),
    p,
    width = 8,
    # width = 4.2,
    height = 2.5
    # height = 1.6
  )
}

################################################################################
################################################################################

probs_true_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-probs.csv"
probs_98_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-98-probs.csv"
probs_96_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-96-probs.csv"
probs_90_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-90-probs.csv"
probs_80_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-80-probs.csv"
probs_60_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-60-probs.csv"
probs_40_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-40-probs.csv"
probs_20_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-20-probs.csv"
probs_empty_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-empty-probs.csv"

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
  ggplot2::facet_wrap(~PromptIndex, nrow = 1) +
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
probs_true_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-probs.csv"
probs_98_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-98-probs.csv"
probs_96_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-96-probs.csv"
probs_90_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-90-probs.csv"
probs_80_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-80-probs.csv"
probs_60_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-60-probs.csv"
probs_40_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-40-probs.csv"
probs_20_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-20-probs.csv"
probs_empty_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-empty-probs.csv"

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
probs_true_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-probs.csv"
probs_98_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-98-probs.csv"
probs_96_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-96-probs.csv"
probs_90_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-90-probs.csv"
probs_80_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-80-probs.csv"
probs_60_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-60-probs.csv"
probs_40_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-40-probs.csv"
probs_20_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-20-probs.csv"
probs_empty_filename <- "results/ml3-gumbel-deletion-10-10-0.0-re-calculated-empty-probs.csv"

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
probs_true_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-probs.csv"
probs_98_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-98-probs.csv"
probs_96_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-96-probs.csv"
probs_90_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-90-probs.csv"
probs_80_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-80-probs.csv"
probs_60_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-60-probs.csv"
probs_40_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-40-probs.csv"
probs_20_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-20-probs.csv"
probs_empty_filename <- "results/ml3-gumbel-deletion-20-20-0.0-re-calculated-empty-probs.csv"

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
  ggplot2::facet_wrap(~variable, scales = "free") +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "bottom")
ggplot2::ggsave("results/probs-diff.pdf", p, width = 5, height = 3)

################################################################################
################################################################################

# The following piece code is no longer used.
theoretical_df_power_i$label_x <- prompt_count + 3
for (llm in unique(theoretical_df_power_i$LLM)) { # nolint
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

interested_metrics <- c(1, 2, 11, 24)
interested_metrics_level <- c(2, 24, 11, 1)
color_palette <- c("baseline", "oracle", "empty", "optim")
names(color_palette) <- interested_metrics
interested_tokens <- c(10, 20, 30)
threshold <- 0.05

for (model_prefix in models_folders_prefix) {
  for (attack in c("substitution")) {
    df_attack <- df[df$Attack == attack & df$Metric %in% interested_metrics, ]
    powers <- cbind(
      Threshold = threshold,
      aggregate(
        df_attack$PValue <= threshold,
        by = list(
          LLM = df_attack$LLM,
          TokensCount = df_attack$TokensCount,
          B = df_attack$B,
          AttackPct = df_attack$AttackPct,
          Metric = df_attack$Metric
        ),
        FUN = function(x) c(Mean = mean(x, na.rm = TRUE), StdError = sd(x, na.rm = TRUE) / sqrt(length(x)))
      )
    )
    powers <- do.call(data.frame, powers)
    powers <- powers[powers$AttackPct > 0, ]
    powers$B <- as.numeric(powers$B)
    powers$TokensCount <- as.numeric(powers$TokensCount)
    powers$k_tokens_count_ratio <- powers$B / powers$TokensCount
    powers <- powers[order(
      powers$k_tokens_count_ratio,
      powers$TokensCount,
      powers$Threshold,
      powers$LLM,
      powers$Metric
    ), ]

    for (k_tokens_count_ratio in k_tokens_count_ratio_list) {
      print(paste(model_prefix, threshold, attack, k_tokens_count_ratio, "power"))
      tab <- NULL
      for (metric in interested_metrics) {
        tab <- rbind(
          tab,
          t(powers[
            powers$LLM == model_prefix &
              powers$Threshold == threshold &
              powers$Metric == metric &
              powers$k_tokens_count_ratio == k_tokens_count_ratio, c("x.Mean", "x.StdError")
          ])
        )
      }
      rownames(tab) <- rep(color_palette, each = 2)
      print(xtable::xtable(tab, type = "latex", digits = 3))
    }
  }
}

for (model_prefix in models_folders_prefix) {
  for (attack in c("deletion", "insertion")) {
    df_attack <- df[df$Attack == attack & df$Metric %in% interested_metrics, ]
    powers <- cbind(
      Threshold = threshold,
      aggregate(
        df_attack$PValue <= threshold,
        by = list(
          LLM = df_attack$LLM,
          TokensCount = df_attack$TokensCount,
          B = df_attack$B,
          Metric = df_attack$Metric
        ),
        FUN = function(x) c(Mean = mean(x, na.rm = TRUE), StdError = sd(x, na.rm = TRUE) / sqrt(length(x)))
      )
    )
    powers <- do.call(data.frame, powers)
    powers$B <- as.numeric(powers$B)
    powers$TokensCount <- as.numeric(powers$TokensCount)
    powers$k_tokens_count_ratio <- powers$B / powers$TokensCount
    powers$Metric <- factor(
      powers$Metric,
      levels = interested_metrics_level
    )

    p <- ggplot2::ggplot(
      powers[
        powers$LLM == model_prefix &
          powers$Threshold == threshold &
          powers$TokensCount %in% interested_tokens,
      ],
      ggplot2::aes(x = TokensCount, y = x.Mean, color = Metric)
    ) +
      ggplot2::geom_line() +
      ggplot2::geom_errorbar(
        ggplot2::aes(ymin = x.Mean - x.StdError, ymax = x.Mean + x.StdError),
        width = 0.5
      ) +
      ggplot2::scale_color_hue(labels = color_palette) +
      ggplot2::facet_grid(
        ~k_tokens_count_ratio,
        labeller = ggplot2::labeller(
          k_tokens_count_ratio = c("0.3" = "B/m: 0.3", "0.6" = "B/m: 0.6", "1" = "B/m: 1.0")
        )
      ) +
      ggplot2::theme_minimal() +
      ggplot2::xlab("Text length") +
      ggplot2::ylab("Power") +
      ggplot2::scale_x_continuous(
        breaks = interested_tokens,
        labels = interested_tokens
      )
    ggplot2::ggsave(
      paste0("results/", model_prefix, "-", attack, ".pdf"),
      p,
      width = 8,
      height = 2.5
    )

    print(paste0("Model: ", model_prefix, ", Attack: ", attack))

    print(
      aggregate(
        df_attack[df_attack$LLM == model_prefix, ]$AttackPct,
        by = list(
          TokensCount = df_attack[df_attack$LLM == model_prefix, ]$TokensCount
        ),
        FUN = function(x) mean(x, na.rm = TRUE)
      )
    )

    powers <- powers[order(
      powers$k_tokens_count_ratio,
      powers$TokensCount,
      powers$Threshold,
      powers$LLM,
      powers$Metric
    ), ]

    for (k_tokens_count_ratio in k_tokens_count_ratio_list) {
      print(paste(model_prefix, threshold, attack, k_tokens_count_ratio, "power"))
      tab <- NULL
      for (metric in interested_metrics) {
        tab <- rbind(
          tab,
          t(powers[
            powers$LLM == model_prefix &
              powers$Threshold == threshold &
              powers$Metric == metric &
              powers$k_tokens_count_ratio == k_tokens_count_ratio, c("x.Mean", "x.StdError")
          ])
        )
      }
      rownames(tab) <- rep(color_palette, each = 2)
      print(xtable::xtable(tab, type = "latex", digits = 3))
    }
  }
}

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

set.seed(1)

folder <- "results/"
models <- c("meta-llama/Meta-Llama-3-8B")
models_folders_prefix <- c("ml3")
generation_methods <- c("gumbel", "transform")
attacks <- c("substitution")
watermark_key_token_pairs <- matrix(c(
  10, 10,
  20, 20,
  30, 30
), ncol = 2, byrow = TRUE)
attack_pcts <- list(
  "deletion" = c("1.0"),
  "insertion" = c("1.0"),
  "substitution" = c("0.0")
)

pvalue_files_templates <- matrix(NA, 0, 17)
for (wkt_index in seq_len(nrow(watermark_key_token_pairs))) { # nolint
  watermark_key_length <- watermark_key_token_pairs[wkt_index, 1]
  tokens_count <- watermark_key_token_pairs[wkt_index, 2]
  max_k <- tokens_count * 1.0
  for (model_index in seq_along(models)) {
    for (generation_methods_index in seq_along(generation_methods)) {
      for (attack_index in seq_along(attacks)) {
        attack_pcts_seq <- attack_pcts[[attacks[attack_index]]]
        for (attack_pct in attack_pcts_seq) {
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
            attack_pct,
            ifelse(generation_methods[generation_methods_index] == "gumbel", "", "-"),
            ifelse(generation_methods[generation_methods_index] == "gumbel", "", max_k),
            ifelse(generation_methods[generation_methods_index] == "gumbel", ".p-detect/", "-detect/"),
            "watermarked-XXX.csv"
          ))
        }
      }
    }
  }
}

prompt_count <- 100
dfs <- list()
filename <- sub("XXX", 0, paste0(pvalue_files_templates[1, ], collapse = ""))

clusters <- parallel::makeCluster(parallel::detectCores() - 1)
doParallel::registerDoParallel(clusters)
for (template_index in seq_len(nrow(pvalue_files_templates))) {
  metric_count <- ifelse(pvalue_files_templates[template_index, 4] == "gumbel", 24, 1)
  print(paste("Processing", template_index, "of", nrow(pvalue_files_templates)))
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
      pvalues_matrix[, prompt_index]
    )
  }
}
parallel::stopCluster(clusters)

df <- do.call(rbind, dfs)
df <- data.frame(df)
names(df) <- c(
  "LLM", "GenerationMethod", "Attack", "WatermarkKeyLength", "TokensCount",
  "AttackPct", "PromptIndex", "Metric", "PValue"
)
df <- as.data.frame(lapply(df, unlist)) # nolint
df$LLM <- as.character(df$LLM)
df$AttackPct <- as.numeric(df$AttackPct)
df$WatermarkKeyLength <- as.numeric(df$WatermarkKeyLength)
df$TokensCount <- as.numeric(df$TokensCount)
df$PromptIndex <- as.numeric(df$PromptIndex)
df$Metric <- as.factor(df$Metric)
df$PValue <- as.numeric(df$PValue)

interested_metrics <- c(1, 2, 11, 24)
threshold <- 0.05

theoretical_df <- data.frame(df[
  df$Attack == "substitution" &
    df$AttackPct == 0,
  c(
    "LLM", "GenerationMethod", "WatermarkKeyLength", "TokensCount",
    "PromptIndex", "Metric", "PValue"
  )
])
theoretical_df_power <- aggregate(
  theoretical_df$PValue <= threshold,
  by = list(
    LLM = theoretical_df$LLM,
    WatermarkKeyLength = theoretical_df$WatermarkKeyLength,
    TokensCount = theoretical_df$TokensCount,
    Metric = theoretical_df$Metric,
    GenerationMethod = theoretical_df$GenerationMethod
  ),
  FUN = function(x) c(Mean = mean(x, na.rm = TRUE), StdError = sd(x, na.rm = TRUE) / sqrt(length(x)))
)
theoretical_df_power <- do.call(data.frame, theoretical_df_power)

theoretical_df_power_i <- theoretical_df_power[
  theoretical_df_power$Metric %in% interested_metrics,
]
theoretical_df_power_i$MetricName <- c(rep("baseline", 3), rep("oracle", 3), rep("empty", 3), rep("optim", 3), rep("ITS", 3))
model_prefix <- "ml3"
theoretical_df_power_i$MetricName <- factor(
  theoretical_df_power_i$MetricName,
  levels = c("oracle", "optim", "empty", "baseline", "ITS")
)

p <- ggplot2::ggplot(
  theoretical_df_power_i[theoretical_df_power_i$TokensCount %in% c(10, 20, 30), ],
  ggplot2::aes(x = TokensCount, y = x.Mean, color = MetricName)
) +
  ggplot2::geom_line() +
  ggplot2::geom_errorbar(
    ggplot2::aes(ymin = x.Mean - x.StdError, ymax = x.Mean + x.StdError),
    width = 0.3
  ) +
  ggplot2::theme_minimal() +
  ggplot2::scale_color_hue(
    labels = c(oracle = "oracle", optim = "optim", empty = "empty", baseline = "baseline", ITS = "ITS"),
    name = "Metric"
  ) +
  ggplot2::xlab("Text length") +
  ggplot2::ylab("Power") +
  ggplot2::scale_x_continuous(
    breaks = c(10, 20, 30),
    labels = c(10, 20, 30)
  )
ggplot2::ggsave(
  paste0("results/theoretical-", model_prefix, "-", threshold, ".pdf"),
  p,
  width = 4.2,
  height = 1.9
)

#  LLM       Attack TokensCount AttackPct  B PromptIndex Metric Criteria PValue

for (model_prefix in models_folders_prefix) {
  plot_df <- df[df$LLM == model_prefix, ]
  plot_df <- aggregate(
    plot_df$PValue <= 0.05,
    by = list(
      TokensCount = plot_df$TokensCount,
      Criteria = plot_df$Criteria
    ),
    FUN = function(x) c(Mean = mean(x, na.rm = TRUE), StdError = sd(x, na.rm = TRUE) / sqrt(length(x)))
  )
  plot_df <- do.call(data.frame, plot_df)
  plot_df$TokensCount <- as.numeric(plot_df$TokensCount)
  plot_df$Criteria <- factor(plot_df$Criteria)

  ggplot2::ggplot(
    plot_df,
    ggplot2::aes(x = TokensCount, y = x.Mean, color = Criteria)
  ) +
    ggplot2::geom_line() +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = x.Mean - x.StdError, ymax = x.Mean + x.StdError),
      width = 0.5
    ) +
    ggplot2::theme_minimal() +
    ggplot2::xlab("Text length") +
    ggplot2::ylab("Type I Error Rate") +
    ggplot2::scale_x_continuous(
      breaks = c(10, 20, 30),
      labels = c(10, 20, 30)
    )
}

df_without_substitution <- df[
  !((df$Attack == "substitution") & (df$AttackPct == 0)) &
    !((df$Attack == "deletion") & (df$Metric == 2)) &
    !((df$Attack == "insertion") & (df$Metric == 2)),
]

powers <- aggregate(
  df_without_substitution$PValue <= 0.05,
  by = list(
    LLM = df_without_substitution$LLM,
    Attack = df_without_substitution$Attack,
    TokensCount = df_without_substitution$TokensCount,
    B = df_without_substitution$B,
    Metric = df_without_substitution$Metric
  ),
  FUN = function(x) c(Mean = mean(x, na.rm = TRUE), StdError = sd(x, na.rm = TRUE) / sqrt(length(x)))
)
powers <- do.call(data.frame, powers)
powers$Attack[powers$Attack == "substitution"] <- "no attack"
powers$Metric[powers$Metric == 1] <- "baseline"
powers$Metric[powers$Metric == 2] <- "oracle"
powers$Metric[powers$Metric == 11] <- "empty"
powers$Metric[powers$Metric == 24] <- "optim"
powers$Metric <- factor(powers$Metric, levels = c("oracle", "optim", "empty", "baseline"))
powers$TokensCount <- as.numeric(powers$TokensCount)
powers$Attack <- factor(powers$Attack, levels = c("no attack", "deletion", "insertion"))

for (model_prefix in models_folders_prefix) {
  powers_llm <- powers[powers$LLM == model_prefix, ]

  ggplot2::ggplot(
    powers_llm,
    ggplot2::aes(x = TokensCount, y = x.Mean, color = Metric)
  ) +
    ggplot2::geom_line() +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = x.Mean - x.StdError, ymax = x.Mean + x.StdError),
      width = 0.2
    ) +
    ggplot2::facet_grid(~Attack) +
    ggplot2::theme_minimal() +
    ggplot2::xlab("Text length") +
    ggplot2::ylab("Power") +
    ggplot2::scale_x_continuous(
      breaks = c(10, 20, 30),
      labels = c(10, 20, 30)
    ) +
    ggplot2::theme(legend.position = "bottom")
  ggplot2::ggsave(
    paste0(model_prefix, "-EMS-power.pdf"),
    width = 8,
    height = 2.5
  )
}

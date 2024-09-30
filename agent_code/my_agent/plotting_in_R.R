# -----------------------------Epislon - Decay
library(ggplot2)
library(readr)
library(scales)

# Read the CSV file, specifying that it has a header
data <- read_csv("epsilons.csv", col_names = TRUE)

# Ensure 'Step' is numeric
data$Step <- as.numeric(data$Step)

# Create the plot
p <- ggplot(data, aes(x = Step, y = Epsilon)) +
  geom_line(color = "#FFB6C1", size = 0.9) +
  labs(title = "Decay of ε in ε-greedy Policy",
       x = "Time (training steps)",
       y = "Exploration Rate (ε)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title = element_text(face = "bold", size = 12),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  scale_x_continuous(labels = comma_format()) +
  scale_y_continuous(labels = number_format(accuracy = 0.01), limits = c(0.85, 0.95))

# Print the plot
print(p)

# Save the plot
ggsave("epsilon_decay_plot.png", plot = p, width = 10, height = 6, dpi = 300, bg = "white")



# ------------------ EPSILON - GREEDY

# Read the CSV file
data <- read_csv("exploration_data.csv")

# Reshape the data from wide to long format
data_long <- pivot_longer(data, cols = c(Exploration, Exploitation), 
                          names_to = "Action_Type", values_to = "Count")

# Create the plot
p <- ggplot(data_long, aes(x = Action_Type, y = Count, fill = Action_Type)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = Count), vjust = -0.5, size = 4) +
  labs(title = "Exploration vs. Exploitation Rate",
       x = "Action Type",
       y = "Count",
       subtitle = paste("Total Steps:", data$Step)) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title = element_text(face = "bold", size = 12),
    legend.position = "none"
  ) +
  scale_fill_manual(values = c("Exploration" = "#FFB6C1", "Exploitation" = "#add8e6"))

# Print the plot
print(p)

# Save the plot
ggsave("exploration_exploitation_plot.png", plot = p, width = 10, height = 6, dpi = 300)


# ---------------- ACTION DISTRIBUTION ------------------------
library(dplyr)

# Read the CSV file
data <- read_csv("action_distribution.csv")

# Reshape the data from wide to long format
data_long <- pivot_longer(data, cols = -Step, names_to = "Action", values_to = "Count")

# Calculate percentages
data_long <- data_long %>%
  mutate(Percentage = Count / sum(Count) * 100)

# Create a custom color palette
custom_colors <- c("UP" = "#add8e6", "DOWN" = "#FFA6C1", "LEFT" = "#adbce6", 
                   "RIGHT" = "#FFB6C1", "WAIT" = "#e78ac3", "BOMB" = "#d8e6ad")

# Create the plot
p <- ggplot(data_long, aes(x = reorder(Action, -Percentage), y = Percentage, fill = Action)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)), vjust = -0.5, size = 4) +
  labs(title = "Action Distribution",
       x = "Action",
       y = "Percentage",
       subtitle = paste("Total Steps:", unique(data$Step))) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title = element_text(face = "bold", size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  ) +
  scale_y_continuous(limits = c(0, max(data_long$Percentage) * 1.1)) +
  scale_fill_manual(values = custom_colors)

# Print the plot
print(p)

# Save the plot
ggsave("action_distribution_plot.png", plot = p, width = 10, height = 6, dpi = 300)

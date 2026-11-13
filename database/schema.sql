CREATE DATABASE IF NOT EXISTS gnn_harassment
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE gnn_harassment;

CREATE TABLE IF NOT EXISTS datasets (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  dataset_name VARCHAR(255) NOT NULL,
  is_demo BOOLEAN NOT NULL DEFAULT FALSE,
  row_count INT NOT NULL,
  user_count INT NOT NULL,
  edge_count INT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_datasets_created (created_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS interactions (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  dataset_id BIGINT NOT NULL,
  source_user VARCHAR(191) NOT NULL,
  target_user VARCHAR(191) NOT NULL,
  message_text TEXT,
  is_abusive TINYINT NULL,
  interaction_time DATETIME NULL,
  interaction_type VARCHAR(80) NULL,
  report_count DOUBLE NOT NULL DEFAULT 0,
  user_label TINYINT NULL,
  CONSTRAINT fk_interactions_dataset FOREIGN KEY (dataset_id)
    REFERENCES datasets(id) ON DELETE CASCADE,
  INDEX idx_interactions_dataset (dataset_id),
  INDEX idx_interactions_source (dataset_id, source_user),
  INDEX idx_interactions_target (dataset_id, target_user),
  INDEX idx_interactions_time (dataset_id, interaction_time)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS model_runs (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  dataset_id BIGINT NOT NULL,
  metrics_json LONGTEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_model_dataset FOREIGN KEY (dataset_id)
    REFERENCES datasets(id) ON DELETE CASCADE,
  INDEX idx_model_dataset (dataset_id)
) ENGINE=InnoDB;

/* deletes the database and creates a new one */


DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS completion;
DROP TABLE IF EXISTS model;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);

CREATE TABLE model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT UNIQUE NOT NULL,
    group_size int,
    num_layers int,
    d_model int,
    dff int,
    num_heads int,
    dropout_rate float,
    learning_rate float,
    batch_size int
    FOREIGN KEY(user_id) REFERENCES user(id)
);


CREATE TABLE completion (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    prompt TEXT NOT NULL,
    answer TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES user(id),
    FOREIGN KEY(model_id) REFERENCES model(id)
);

INSERT INTO model 
  (user_id, model_name, num_layers, num_heads), 
    VALUES (1, 'bloom', 70, 112, 14336)
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
    user_id INTEGER NOT NULL,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT UNIQUE NOT NULL,
    set_size int NOT NULL,
    num_layers int NOT NULL,
    d_model int NOT NULL,
    dff int NOT NULL,
    num_heads int NOT NULL,
    dropout_rate float NOT NULL,
    learning_rate float NOT NULL,
    batch_size int NOT NULL,
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


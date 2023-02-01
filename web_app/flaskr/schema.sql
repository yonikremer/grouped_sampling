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
    model_name TEXT UNIQUE NOT NULL,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES user(id)
);


CREATE TABLE completion (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    group_size INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    answer TEXT NOT NULL,
    num_tokens INTEGER NOT NULL,
    generation_type TEXT NOT NULL,
    top_p FLOAT,
    top_k INTEGER,
    temperature FLOAT DEFAULT 1,
    FOREIGN KEY(user_id) REFERENCES user(id),
    FOREIGN KEY(model_id) REFERENCES model(id)
);

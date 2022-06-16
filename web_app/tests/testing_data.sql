-- noinspection SpellCheckingInspectionForFile

INSERT INTO user (username, password)
VALUES
  ('test', 'pbkdf2:sha256:50000$TCI4GzcX$0de171a4f4dac32e3364c7ddc7c14f3e2fa61f2d17574483f7ffbb431b4acb2f'),
  ('other', 'pbkdf2:sha256:50000$kJPKsz6N$d2d4784f1b030a9761f5ccaeeaca413f27f2ecb76d6168407af962ddce849f79'),
  ('admin', 'pbkdf2:sha256:50000$kJPKsz6N$d2d4784f1b030a9761f5ccaeeaca413f27f2ecb76d6168407af962ddce849f79'),
  ('test2', 'pbkdf2:sha256:50000$TCI4GzcX$0de171a4f4dac32e3364c7ddc7c14f3e2fa61f2d17574483f7ffbb431b4acb2f');

INSERT INTO completion (user_id, model_id, created, prompt, answer)
VALUES
  (1, 1, '2018-01-01 00:00:00', 'Is this a test?', 'Of course it is just a test!'),
  (2, 1, '2018-01-01 00:00:00', 'The porpuse of testing', 'Is to assert everything works ass intended');


INSERT INTO model (user_id, created, model_name, set_size, num_layers, d_model, dff, num_heads, dropout_rate, learning_rate, batch_size)
VALUES
  (1, '2018-08-01 11:00:00', 'test_model1', 128, 2, 128, 512, 16, 0.1, 0.001, 32),
  (2, '2022-01-01 00:00:00', 'test_model2', 64, 3, 128, 512, 32, 0.15, 0.0005, 32),
  (3, '2019-01-01 00:10:59', 'test_model3', 256, 4, 128, 512, 8, 0.075, 0.0001, 64);
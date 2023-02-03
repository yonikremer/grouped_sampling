-- noinspection SpellCheckingInspectionForFile

INSERT INTO user (username, password) VALUES
('test',
'pbkdf2:sha256:50000$TCI4GzcX$0de171a4f4dac32e3364c7ddc7c14f3e2fa61f2d17574483f7ffbb431b4acb2f'),
('other',
'pbkdf2:sha256:50000$kJPKsz6N$d2d4784f1b030a9761f5ccaeeaca413f27f2ecb76d6168407af962ddce849f79'),
('admin',
'pbkdf2:sha256:50000$kJPKsz6N$d2d4784f1b030a9761f5ccaeeaca413f27f2ecb76d6168407af962ddce849f79'),
('test2',
'pbkdf2:sha256:50000$TCI4GzcX$0de171a4f4dac32e3364c7ddc7c14f3e2fa61f2d17574483f7ffbb431b4acb2f');

INSERT INTO completion (user_id, prompt, answer, num_tokens, group_size, model_id, generation_type) VALUES
(1,
'Is this a test?',
'Is this a test? Of course it is just a test!',
7, 1, 2,
'tree'),
(2,
'The porpuse of testing',
'The porpuse of testing Is to assert everything works ass intended',
8, 2, 3, 'sampling');


INSERT INTO model (user_id, model_name) VALUES
(1, 'test_model1'),
(2, 'test_model2'),
(3, 'test_model3');

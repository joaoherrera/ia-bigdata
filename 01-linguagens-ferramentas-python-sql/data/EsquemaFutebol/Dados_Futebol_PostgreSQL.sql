insert into time values ('Sao Paulo',     'SP', 'profissional', 8);
insert into time values ('Palmeiras',     'SP', 'profissional', 5);
insert into time values ('Santos',        'SP', 'profissional', 0);
insert into time values ('Corinthians',   'SP', 'profissional', 6);
insert into time values ('Paulistinha',   'SP', 'amador',       1);
insert into time values ('Ibate',         'SP', 'amador',       0);
insert into time values ('Cruzeiro',      'MG', 'profissional', 2);
insert into time values ('Atletico',      'MG', 'profissional', 3);
insert into time values ('Frutal',        'MG', 'amador',       1);

insert into joga values ('Sao Paulo',   'Palmeiras',   'S');
insert into joga values ('Ibate',       'Paulistinha', 'N');
insert into joga values ('Ibate',       'Frutal',      'S');
insert into joga values ('Santos',      'Corinthians', 'N');
insert into joga values ('Corinthians', 'Palmeiras',   'S');
insert into joga values ('Santos',      'Sao Paulo',    'N');
insert into joga values ('Sao Paulo',   'Corinthians', 'S');
insert into joga values ('Santos',      'Palmeiras',   'N');

insert into partida values ('Sao Paulo', 'Palmeiras', '2007-05-15', '2x0');
insert into partida values ('Santos', 'Sao Paulo',   '2007-05-20', 'Pacaembu',   '1x0');
insert into partida values ('Santos',    'Palmeiras',  '2007-06-06', 'Vila Belmiro', '0x1');
insert into partida (time1, time2, data, local) values ('Ibate','Paulistinha', '2007-05-25', 'Luizao');
insert into partida (time1, time2, data, local) values ('Santos',    'Corinthians', '2007-05-30', 'Pacaembu');

insert into jogador values (111111111, 'Pele', '1995-05-15', 'Santos', 'Santos');
insert into jogador values (111111112, 'Garrincha', '1945-12-01', 'Sao Paulo', 'Ibate');
insert into jogador values (111111113, 'Muller', '1960-09-09', 'Sao Paulo', 'Sao Paulo');
insert into jogador values (111111114, 'Zeca',  '1980-09-09', 'Frutal', 'Frutal');
insert into jogador (rg, nome, data_nascimento, naturalidade) values (111111115, 'Juca', '1982-09-09', 'Sao Carlos');


insert into posicao_jogador values (111111113, 'Atacante');
insert into posicao_jogador values (111111112, 'Zagueiro');
insert into posicao_jogador values (111111111, 'Centroavante');

insert into diretor values ('Sao Paulo',   'Joao',    'da Silva');
insert into diretor values ('Palmeiras',   'Jose',    'Souza');
insert into diretor values ('Corinthians', 'Tome',    'Sauza');
insert into diretor values ('Ibate',       'Manuel',  'Carlos');
insert into diretor values ('Paulistinha', 'Joaquim', 'Moura');

insert into uniforme values ('Atletico', 'titular', 'vermelho');
insert into uniforme values ('Atletico', 'reserva', 'preto');
insert into uniforme values ('Sao Paulo',   'titular', 'vermelho');
insert into uniforme values ('Sao Paulo',   'reserva', 'branca');
insert into uniforme values ('Palmeiras',   'titular', 'verde');
insert into uniforme values ('Palmeiras',   'reserva', 'branca');
insert into uniforme values ('Corinthians', 'titular', 'branca');
insert into uniforme values ('Corinthians', 'reserva', 'preto');
insert into uniforme values ('Frutal', 'titular', 'vermelho');
insert into uniforme values ('Frutal', 'reserva', 'preto');

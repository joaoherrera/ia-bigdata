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

insert into partida values ('Sao Paulo', 'Palmeiras', TO_DATE('15/05/2007', 'DD/MM/YYYY'), 'Morumbi', '2x0');
insert into partida values ('Santos', 'Sao Paulo',   TO_DATE('20/05/2007', 'DD/MM/YYYY'), 'Pacaembu',   '1x0');
insert into partida values ('Santos',    'Palmeiras',  TO_DATE('06/06/2007', 'DD/MM/YYYY'), 'Vila Belmiro', '0x1');
insert into partida (time1, time2, data, local) values ('Ibate','Paulistinha', TO_DATE('25/05/2007', 'DD/MM/YYYY'), 'Luizao');
insert into partida (time1, time2, data, local) values ('Santos',    'Corinthians', TO_DATE('30/05/2007', 'DD/MM/YYYY'), 'Pacaembu');

insert into jogador values (111111111, 'Pele', TO_DATE('15/05/1955', 'DD/MM/YYYY'), 'Santos', 'Santos');
insert into jogador values (111111112, 'Garrincha', TO_DATE('01/12/1945', 'DD/MM/YYYY'), 'Sao Paulo', 'Ibate');
insert into jogador values (111111113, 'Muller', TO_DATE('09/09/1960', 'DD/MM/YYYY'), 'Sao Paulo', 'Sao Paulo');
insert into jogador values (111111114, 'Zeca',  TO_DATE('09/09/1980', 'DD/MM/YYYY'), 'Frutal', 'Frutal');
insert into jogador (rg, nome, data_nascimento, naturalidade) values (111111115, 'Juca',  TO_DATE('09/09/1982', 'DD/MM/YYYY'), 'Sao Carlos');


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

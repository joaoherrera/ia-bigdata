Drop table time cascade constraints;
Drop table joga cascade constraints;
Drop table partida cascade constraints;
Drop table jogador cascade constraints;
Drop table posicao_jogador cascade constraints;
Drop table diretor cascade constraints;
Drop table uniforme cascade constraints;


CREATE TABLE time (
      nome       VARCHAR2(40) NOT NULL,
      estado     CHAR(2),
      tipo       VARCHAR2(15),
      saldo_gols INTEGER,
      
      CONSTRAINT pk_time PRIMARY KEY (nome),
      CONSTRAINT ck_tipo CHECK (tipo in ('amador', 'profissional'))
);

CREATE TABLE joga (
      time1      VARCHAR2(40) NOT NULL,
      time2      VARCHAR2(40) NOT NULL,
      classico   CHAR(1),

      CONSTRAINT pk_joga PRIMARY KEY (time1, time2),
      CONSTRAINT fk_joga1 FOREIGN KEY (time1) REFERENCES time(NOME) ON DELETE CASCADE,
      CONSTRAINT fk_joga2 FOREIGN KEY (time2) REFERENCES time(NOME) ON DELETE CASCADE,
      CONSTRAINT ck_classico CHECK (classico in ('S', 'N'))
);

CREATE TABLE partida (
      time1      VARCHAR2(40) NOT NULL,
      time2      VARCHAR2(40) NOT NULL,
      data       DATE NOT NULL,
      local      VARCHAR2(40),
      placar     VARCHAR2(5) DEFAULT '0x0' NOT NULL,

      CONSTRAINT pk_partida PRIMARY KEY (time1, time2, data),
      CONSTRAINT fk_partida FOREIGN KEY (time1, time2) REFERENCES joga(time1, time2) ON DELETE CASCADE,
      CONSTRAINT ck_placar  CHECK (placar like '%x%')
);

CREATE TABLE jogador (
      rg              VARCHAR2(15) NOT NULL,
      nome            VARCHAR2(40) NOT NULL,
      data_nascimento DATE,
      naturalidade    VARCHAR2(40),
      time_atua       VARCHAR2(40),

      CONSTRAINT pk_jogador PRIMARY KEY (rg),
      CONSTRAINT un_jogador UNIQUE (nome),
      CONSTRAINT fk_jogador FOREIGN KEY (time_atua) REFERENCES time(nome) ON DELETE SET NULL
);

CREATE TABLE posicao_jogador (
      jogador         VARCHAR2(15) NOT NULL,
      posicao         VARCHAR2(40) NOT NULL,

      CONSTRAINT pk_posicao_jogador PRIMARY KEY (jogador, posicao),
      CONSTRAINT fk_posicao_jogador FOREIGN KEY (jogador) REFERENCES jogador(rg) ON DELETE CASCADE
);

CREATE TABLE diretor (
      timedir         VARCHAR2(40) NOT NULL,
      nome            VARCHAR2(50) NOT NULL,
      sobrenome       VARCHAR2(50) NOT NULL,

      CONSTRAINT pk_diretor PRIMARY KEY (timedir, nome, sobrenome),
      CONSTRAINT fk_diretor FOREIGN KEY (timedir) REFERENCES time(nome) ON DELETE CASCADE
);

CREATE TABLE uniforme (
      timeunif        VARCHAR2(40) NOT NULL,
      tipo            VARCHAR2(10) NOT NULL,
      corprincipal    VARCHAR2(30) NOT NULL,

      CONSTRAINT pk_uniforme PRIMARY KEY (timeunif, tipo),
      CONSTRAINT fk_uniforme FOREIGN KEY (timeunif) REFERENCES time(nome) ON DELETE CASCADE,
      CONSTRAINT ck_uniforme CHECK (tipo in ('titular', 'reserva'))
);

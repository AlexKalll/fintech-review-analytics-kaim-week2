ALTER SESSION SET CONTAINER = XEPDB1;
CREATE USER bank_reviews IDENTIFIED BY Welcome123 QUOTA UNLIMITED ON USERS;
GRANT CONNECT, RESOURCE, CREATE SESSION, CREATE TABLE, CREATE SEQUENCE TO bank_reviews;

CREATE TABLE Banks (
    bank_id NUMBER PRIMARY KEY,
    bank_name VARCHAR2(100) NOT NULL
);

CREATE TABLE Reviews (
    review_id NUMBER PRIMARY KEY,
    bank_id NUMBER,
    review VARCHAR2(4000),
    rating NUMBER(2,1),
    review_date DATE,
    source VARCHAR2(100),
    sentiment VARCHAR2(50),
    confidence NUMBER(3,2),
    theme VARCHAR2(100),
    topic_probability NUMBER(3,2),
    FOREIGN KEY (bank_id) REFERENCES Banks(bank_id)
);

CREATE SEQUENCE review_id_seq START WITH 1 INCREMENT BY 1;

INSERT INTO Banks (bank_id, bank_name) VALUES (1, 'Commercial Bank of Ethiopia');
INSERT INTO Banks (bank_id, bank_name) VALUES (2, 'Bank of Abyssinia');
INSERT INTO Banks (bank_id, bank_name) VALUES (3, 'Dashen Bank');
COMMIT;
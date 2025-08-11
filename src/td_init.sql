
-- Create 'amurd' database
CREATE DATABASE AMURD AS
    PERM = 2000000*(HASHMAP()+1),
    SPOOL = 2000000*(HASHMAP()+1),
    TERMPORARY = 2000000*(HASHMAP()+1)

-- Create products table
CREATE MULTISET TABLE amurd.products (
    product_id INTEGER NOT NULL,
    product_name VARCHAR(255),
    brand_name VARCHAR(255),
) PRIMARY KEY (product_id);

-- Create classes table
CREATE MULTISET TABLE amurd.classes (
    class_id INTEGER NOT NULL,
    class_name VARCHAR(255),
) PRIMARY KEY (class_id);

-- Create product_embeddings table
CREATE MULTISET TABLE amurd.product_embeddings (
    product_id INTEGER NOT NULL,
    embeddings BLOB(5000),
) PRIMARY KEY (product_id);

-- Create class_embeddings table
CREATE MULTISET TABLE amurd.class_embeddings (
    class_id INTEGER NOT NULL,
    embeddings BLOB(5000),
) PRIMARY KEY (class_id);
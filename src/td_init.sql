 
-- Create 'amurd' database
CREATE DATABASE AMURD AS
    PERM = 2000000,
    SPOOL = 2000000,
    TEMPORARY = 2000000;

-- Create products table
CREATE MULTISET TABLE amurd.products (
    product_id INTEGER NOT NULL,
    product_name VARCHAR(1024),
    brand_name VARCHAR(1024)
) PRIMARY INDEX (product_id);

-- Create classes table
CREATE MULTISET TABLE amurd.classes (
    class_id INTEGER NOT NULL,
    class_name VARCHAR(1024),
    class_description VARCHAR(1024)
) PRIMARY INDEX (class_id);

-- Create product_embeddings table
CREATE MULTISET TABLE amurd.product_embeddings (
    product_id INTEGER NOT NULL,
    embeddings JSON
) PRIMARY INDEX (product_id);

-- Create class_embeddings table
CREATE MULTISET TABLE amurd.class_embeddings (
    class_id INTEGER NOT NULL,
    embeddings JSON
) PRIMARY INDEX (class_id);
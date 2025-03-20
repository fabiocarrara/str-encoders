# StrLucene

**StrLucene** is a simple Lucene-based ANN search engine for indexing and searching surrogate-text representation of dense vectors.

## Getting Started
Build the JAR file with Maven:
```bash
mvn clean compile assembly:single
```

## Usage

**StrLucene** is stdin/stdout based. Each line of stdin is either a document to be indexed or a query to be searched, for the indexing and search procedures respectively. Look at `generate_strs.py` script for an example of how to generate docs and queries to be used with **StrLucene**.

### Index
Each line of stdin is treated as a document to be indexed. For each indexed document, a 0-based incremental integer ID is assigned and stored in the index, depending on the order of the documents in the input stream.
```bash
cat test_str_db.txt | java -jar target/StrLucene-1.0.jar index test_index/
```

### Show terms|freqs of an indexed document
For a specified documents, this print the given ID, the internal Lucene ID, and the terms and their frequencies in the document.
```bash
java -jar target/StrLucene-1.0.jar show test_index/ -d 0
```

### Search
Each line of stdin is treated as a query to be searched.
```bash
cat test_str_query.txt | java -jar target/StrLucene-1.0.jar search -k 100 test_index/
```
each line of output is the response to the corresponding query, in this format:
```
<query_time_in_ms>ms <ID1>:<score1> <ID2>:<score2> ... <IDk>:<scorek>
```
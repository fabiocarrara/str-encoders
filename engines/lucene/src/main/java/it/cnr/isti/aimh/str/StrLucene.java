package it.cnr.isti.aimh.str;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.StringJoiner;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Analyzer.TokenStreamComponents;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.miscellaneous.DelimitedTermFrequencyTokenFilter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.ConcurrentMergeScheduler;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.search.similarities.SimilarityBase;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import me.tongfei.progressbar.ProgressBar;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.Namespace;
import net.sourceforge.argparse4j.inf.Subparser;
import net.sourceforge.argparse4j.inf.Subparsers;

public class StrLucene {

    // custom analyzer that parses "term1|freq1 term2|freq2 ..."
    public static class TermFrequencyAnalyzer extends Analyzer {
        @Override
        protected TokenStreamComponents createComponents(String fieldName) {
            Tokenizer source = new WhitespaceTokenizer();
            TokenStream result = new DelimitedTermFrequencyTokenFilter(source);
            return new TokenStreamComponents(source, result);
        }
    }

    // inner product similarity (query term frequencies are expressed as terms
    // boosts)
    public static class InnerProduct extends SimilarityBase {
        @Override
        protected double score(BasicStats stats, double freq, double docLen) {
            return stats.getBoost() * freq;
        }

        @Override
        public String toString() {
            return "InnerProduct()";
        }
    }

    public static void main(String[] args) throws Exception {

        ArgumentParser parser = ArgumentParsers.newFor("StrLucene").build()
                .defaultHelp(true)
                .description("Index and Search Surrogate Text Representations.");

        parser.addArgument("-q", "--quiet")
                .action(Arguments.storeTrue())
                .help("suppress progress bar");

        Subparsers subparsers = parser.addSubparsers().help("sub-command help");

        // index command
        Subparser indexParser = subparsers
                .addParser("index")
                .setDefault("command", "index")
                .help("create an index");

        indexParser
                .addArgument("index-dir")
                .help("directory in which the index is created");

        // show command
        Subparser showParser = subparsers
                .addParser("show")
                .setDefault("command", "show")
                .help("show contents of an index");

        showParser
                .addArgument("index-dir")
                .help("directory of the index to be shown");

        showParser
                .addArgument("-d", "--doc-id")
                .type(Integer.class)
                .setDefault(-1)
                .help("id of document to show");

        // search command
        Subparser searchParser = subparsers
                .addParser("search")
                .setDefault("command", "search")
                .help("search an index");

        searchParser
                .addArgument("index-dir")
                .help("directory of the index to be searched");

        searchParser
                .addArgument("-k", "--num-nearest-neighbors")
                .type(Integer.class)
                .setDefault(100)
                .help("number of nearest neighbors to return per query");

        Namespace ns = parser.parseArgsOrFail(args);

        String subCommand = ns.getString("command");
        switch (subCommand) {
            case "index":
                index(ns);
                break;

            case "show":
                show(ns);
                break;

            case "search":
                search(ns);
                break;

            default:
                parser.printUsage();
                System.exit(1);
        }
    }

    private static void index(Namespace ns) throws IOException {
        // open the index dir
        String indexDir = ns.getString("index_dir");
        Path indexPath = Paths.get(indexDir, "");
        FSDirectory index = FSDirectory.open(indexPath);

        // configure index writer
        Analyzer analyzer = new TermFrequencyAnalyzer(); // parses "term1|freq1 term2|freq2 ..."
        IndexWriterConfig conf = new IndexWriterConfig(analyzer);
        conf.setOpenMode(OpenMode.CREATE);
        conf.setRAMBufferSizeMB(1024);
        conf.setUseCompoundFile(false);
        conf.setMergeScheduler(new ConcurrentMergeScheduler());

        IndexWriter writer = new IndexWriter(index, conf);

        // define schema/fields
        FieldType strFieldType = new FieldType(TextField.TYPE_NOT_STORED);
        strFieldType.setOmitNorms(true);
        strFieldType.setIndexOptions(org.apache.lucene.index.IndexOptions.DOCS_AND_FREQS);

        // read documents from stdin
        BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
        String line;
        int lineCount = 0;

        // progress bar (if not quiet)
        Boolean isQuiet = ns.getBoolean("quiet");
        ProgressBar progress = null;
        if (!isQuiet)
            progress = new ProgressBar("Indexing", -1);

        // index documents (one per line)
        while ((line = stdin.readLine()) != null) {
            Document doc = new Document();
            doc.add(new StoredField("id", lineCount));
            doc.add(new Field("surrogate_text", line, strFieldType));
            writer.addDocument(doc);

            if (!isQuiet)
                progress.step();
            lineCount++;
        }

        // commit to disk (might take a while for large indexes)
        writer.commit();
        writer.close();

        if (!isQuiet)
            progress.close();
    }

    private static void show(Namespace ns) throws IOException {
        // params
        String indexDir = ns.getString("index_dir");
        int docId = ns.getInt("doc_id");

        // open index dir
        Path indexPath = Paths.get(indexDir, "");
        FSDirectory index = FSDirectory.open(indexPath);
        DirectoryReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);
        StoredFields storedFields = searcher.storedFields();

        // no doc id specified, print only stats
        if (docId == -1) {
            System.out.println("Number of documents: " + reader.numDocs());
            return;
        }

        // print the terms and freqs of specified document
        int givenId = storedFields.document(docId).getField("id").numericValue().intValue();
        System.out.print("doc=" + docId + " id=" + givenId + ": ");

        // iterate over segments and print terms and freqs
        reader.leaves().stream()
                .filter(context -> docId >= context.docBase && docId < context.docBase + context.reader().maxDoc())
                .forEach(context -> {
                    try {
                        Terms terms = context.reader().terms("surrogate_text");
                        if (terms == null) {
                            System.err.println("Field 'surrogate_text' does not exist in this segment.");
                            return;
                        }

                        TermsEnum termsEnum = terms.iterator();
                        BytesRef term;
                        PostingsEnum postings = null;

                        while ((term = termsEnum.next()) != null) {
                            postings = termsEnum.postings(postings, PostingsEnum.FREQS);
                            if (postings.advance(docId) == docId) {
                                String termText = term.utf8ToString();
                                int freq = postings.freq();
                                System.out.print(termText + "|" + freq + " ");
                            }
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
        System.out.println();
    }

    private static void search(Namespace ns) throws IOException, ParseException {
        // params
        String indexDir = ns.getString("index_dir");
        int k = ns.getInt("num_nearest_neighbors");

        // open index dir
        Path indexPath = Paths.get(indexDir, "");
        FSDirectory index = FSDirectory.open(indexPath);
        DirectoryReader indexReader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(indexReader);
        StoredFields storedFields = searcher.storedFields();

        // configure similarity and query parser
        Similarity similarity = new InnerProduct();
        searcher.setSimilarity(similarity);

        QueryParser parser = new QueryParser("surrogate_text", new TermFrequencyAnalyzer());

        // read queries from stdin
        BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
        String line;

        // progress bar (if not quiet)
        Boolean isQuiet = ns.getBoolean("quiet");
        ProgressBar progress = null;
        if (!isQuiet)
            progress = new ProgressBar("Searching", -1);

        // search queries (one per line)
        while ((line = stdin.readLine()) != null) {
            // query time = parsing + searching + retrieving fields
            long elapsed = -System.currentTimeMillis();
            Query q = parser.parse(line);
            TopDocs hits = searcher.search(q, k);

            StringJoiner joiner = new StringJoiner(" ");
            for (ScoreDoc hit : hits.scoreDocs) {
                int givenId = storedFields.document(hit.doc).getField("id").numericValue().intValue();
                joiner.add(givenId + ";" + hit.score);
            }
            elapsed += System.currentTimeMillis();

            // print query time and results
            System.out.print(elapsed + "ms ");
            System.out.println(joiner.toString());

            if (!isQuiet)
                progress.step();
        }

        if (!isQuiet)
            progress.close();
    }
}

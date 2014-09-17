package com.vikasing.lsa;

import org.jblas.DoubleMatrix;
import org.jblas.Singular;

import java.util.*;

/**
 * Created by vik on 9/8/14.
 */
public class DoLSA {
    public static void main(String[] args)
    {
        r();
    }
    private static void r() {
        Map<String, Integer> termFrequencyMap = new HashMap<>();
        List<String> docs = Utils.getFileLines("data/good-messages.txt", true);
        List<String[]> tokenizedDocs =  new ArrayList<>();
        // get unique terms
        for (String doc : docs) {
            doc = doc.toLowerCase();
            doc = Utils.removeSpecialChars(doc);
            doc = Utils.removeStopWords(doc);
            String terms[] = doc.split(" ");
            tokenizedDocs.add(terms);
            for (String term : terms) {
                if (term.isEmpty() || term.length() < 2) continue;
                if (termFrequencyMap.containsKey(term)) {
                    termFrequencyMap.put(term, termFrequencyMap.get(term) + 1);
                } else {
                    termFrequencyMap.put(term, 1);
                }
            }
        }
        //keeps only frequent terms
        List<String> frequentTermsList = new ArrayList<>();
        Set<String> termsSet = termFrequencyMap.keySet();
        termsSet.stream().filter(s -> termFrequencyMap.get(s) > 1.0).forEach(frequentTermsList::add);
        String[] sortedTermArray = frequentTermsList.toArray(new String[frequentTermsList.size()]);
        Arrays.sort(sortedTermArray);

        // generate terms x document matrix, document is the column vector
        int numOfTerms = sortedTermArray.length;
        int numOfDocs = docs.size();
        System.out.println("Number of Docs: "+numOfDocs+" Number of Terms: "+ numOfTerms);
        double[][] docTermMatrix = new double[numOfTerms][numOfDocs];
        for (int d = 0; d < numOfDocs; d++) {
            String[] tokens = tokenizedDocs.get(d);
            for (String token : tokens) {
                int pos = Arrays.binarySearch(sortedTermArray,token);
                if (pos>-1)
                docTermMatrix[pos][d] = docTermMatrix[pos][d] + 1.0;
            }
        }

        /*double s[][] = new double[8][5];
        s[0][0] = 1.0;
        s[1][0] = 1.0;
        s[1][1] = 1.0;
        s[2][1] = 1.0;
        s[3][1] = 1.0;
        s[3][2] = 1.0;
        s[4][3] = 1.0;
        s[5][2] = 1.0;
        s[5][3] = 1.0;
        s[6][3] = 1.0;
        s[7][3] = 1.0;
        s[7][4] = 1.0;
        DoubleMatrix doubleMatrix[] = Singular.fullSVD(new DoubleMatrix(s));*/
        DoubleMatrix[] doubleMatrix = Singular.sparseSVD(new DoubleMatrix(docTermMatrix));
        DoubleMatrix S = doubleMatrix[0];
        DoubleMatrix sigma = doubleMatrix[1];
        //DoubleMatrix U = doubleMatrix[2];
        // reducing sigma to a size k
        List<Double> nonZero = new ArrayList<>();
        for (int i = 0; i < sigma.length; i++) {
            if (sigma.get(i) > 0.0) {
                nonZero.add(i, sigma.get(i));
            } else {
                break;
            }
        }
        int k = nonZero.size();
        DoubleMatrix reducedSigma = new DoubleMatrix(nonZero.size());
        for (int i = 0; i < k; i++) {
            reducedSigma.put(i, 0, nonZero.get(i));
        }
        DoubleMatrix reducedS = S.getRange(0,S.rows,0,k);
        //DoubleMatrix reducedU = U.getRange(0,k,0,U.columns);
        Map<String, DoubleMatrix> finalTermVector = new HashMap<>();
        for (int i = 0; i < numOfTerms; i++) {
            finalTermVector.put(sortedTermArray[i], reducedS.getRow(i).mul(reducedSigma));
        }
        String trm = "jessica";
        Map<String, Double> similarityMap = new HashMap<>();
        for (String term : finalTermVector.keySet()) {
            DoubleMatrix main = finalTermVector.get(trm), inQ = finalTermVector.get(term);
            double magni = getMagnitude(inQ.data) * getMagnitude(main.data);
            if (magni > 0.001) {
                similarityMap.put(term, main.dot(inQ) / magni);
            }
        }
        System.out.println(entriesSortedByValues(similarityMap));
    }

    private static double getMagnitude(double[] vector) {
        double mag = 0.0;
        for (int i = 0; i < vector.length; i++) {
            mag += Math.pow(vector[i], 2);
        }
        return Math.sqrt(mag);
    }

    public static <K, V extends Comparable<? super V>> SortedSet<Map.Entry<K, V>> entriesSortedByValues(Map<K, V> map) {
        SortedSet<Map.Entry<K, V>> sortedEntries = new TreeSet<>(
                (e1, e2) -> {
                    int res = e2.getValue().compareTo(e1.getValue());
                    return res != 0 ? res : 1;
                }
        );
        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }
}

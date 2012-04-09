package bit.lin.pairwise.myrank;

public interface SubClassifier {
	void setInputPatterns4t(String[] pair);

	void train();

	boolean classify(String doc1, String doc2);
	
	boolean isRunning();
	
}

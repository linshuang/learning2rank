package bit.lin.pairwise.sortnet;

import java.io.File;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.Reader;
import java.util.StringTokenizer;

import org.joone.io.PatternTokenizer;
import org.joone.io.StreamInputTokenizer;
import org.joone.log.ILogger;
import org.joone.log.LoggerFactory;

/**
 * this class is from StreamInputTokenizer
 */
public class StreamInputTokenizer4SN implements PatternTokenizer {
	/**
	 * Logger
	 * */
	private static final ILogger log = LoggerFactory
			.getLogger(StreamInputTokenizer.class);

	private static final int MAX_BUFFER_SIZE = 1048576; // 1 MByte
	private LineNumberReader stream1;
	private LineNumberReader stream2, stream2Clone;
	private StringTokenizer tokenizer = null;
	private int numTokens = 0;
	private char m_decimalPoint = '.';
	private String m_delim = "; \t\n\r\f";
	private double[] tokensArray;
	private int maxBufSize;

	/**
	 * Creates new StreamInputTokenizer
	 * 
	 * @param in
	 *            The input stream
	 */
	public StreamInputTokenizer4SN(Reader in1, Reader in2)
			throws java.io.IOException {
		this(in1, in2, MAX_BUFFER_SIZE);
	}

	/**
	 * Creates new StreamInputTokenizer
	 * 
	 * @param in
	 *            The input stream
	 * @param maxBufSize
	 *            the max dimension of the input buffer
	 */
	public StreamInputTokenizer4SN(Reader in1, Reader in2, int maxBufSize)
			throws java.io.IOException {
		this.maxBufSize = maxBufSize;
		stream1 = new LineNumberReader(in1, maxBufSize);
		stream1.mark(maxBufSize);
		stream2 = new LineNumberReader(in2, maxBufSize);
		stream2.mark(maxBufSize);
		_line1 = stream1.readLine();
	}

	/**
	 * Return the current line number.
	 * 
	 * @return the current line number
	 */
	public int getLineno() {
		return stream1.getLineNumber();
	}

	public int getNumTokens() throws java.io.IOException {
		return numTokens;
	}

	/**
	 * Insert the method's description here. Creation date: (17/10/2000 0.30.08)
	 * 
	 * @return float
	 * @param posiz
	 *            int
	 */
	public double getTokenAt(int posiz) throws java.io.IOException {
		if (tokensArray == null)
			if (!nextLine())
				return 0;
		if (tokensArray.length <= posiz)
			return 0;
		return tokensArray[posiz];
	}

	/**
	 * Insert the method's description here. Creation date: (17/10/2000 0.13.45)
	 * 
	 * @return float[]
	 */
	public double[] getTokensArray() {
		return tokensArray;
	}

	/**
	 * mark the current position.
	 */
	public void mark() throws java.io.IOException {
		stream1.mark(maxBufSize);
	}

	String _line1, _line2 = "";
	boolean _f1 = true;
	boolean _f2 = true;

	/**
	 * Fetchs the next line and extracts all the tokens
	 * 
	 * @return false if EOF, otherwise true
	 * @throws IOException
	 *             if an I/O Error occurs
	 */
	public boolean nextLine() throws java.io.IOException {
		String inputPattern;
		// String inputPattern4Debug;
		// f2表示是不是需要读文件，还是自动变换
		if (_f2) {
			// find the pair
			findPair(false);
			if (_line1 == null || _line2 == null)
				return false;
			inputPattern = _line1.trim() + ";" + _line2.trim()
					+ getTarget(_line1, _line2);
			
//			System.out.format(
//					"Input pattern#%d : relevance %s - relevance %s\n", i,
//					_line1.split(";")[0], _line2.split(";")[0]);
//			i++;
			
			// inputPattern4Debug = _line1.trim() + "\n" + _line2.trim()
			// + getTarget(_line1, _line2);
			_f2 = false;
		} else {
			inputPattern = _line2.trim() + ";" + _line1.trim()
					+ getTarget(_line2, _line1);
			
//			System.out.format(
//					"Input pattern#%d : relevance %s - relevance %s\n", i,
//					_line2.split(";")[0], _line1.split(";")[0]);
//			i++;
			
			// inputPattern4Debug = _line2.trim() + "\n" + _line1.trim()
			// + getTarget(_line2, _line1);
			_f2 = true;
		}
		// System.out.println(inputPattern4Debug);
		tokenizer = new StringTokenizer(inputPattern, m_delim, false);
		numTokens = tokenizer.countTokens();
		if (tokensArray == null)
			tokensArray = new double[numTokens];
		else if (tokensArray.length != numTokens)
			tokensArray = new double[numTokens];
		for (int i = 0; i < numTokens; ++i)
			tokensArray[i] = nextToken(m_delim);

		return true;
	}

	private String getTarget(String line1, String line2) {
		String rvl1 = line1.split(";")[0], rvl2 = line2.split(";")[0];
		if (new Integer(rvl1) > new Integer(rvl2)) {
			return ";1;0";
		} else
			return ";0;1";
	}

	int i = 1;

	/**
	 * flag用来表示是不是该记录stream2的位置了
	 */
	private void findPair(boolean flag) throws IOException {
		_line2 = stream2.readLine();
		if (_line1 == null)
			return;
		if (_line2 == null) {
			stream2.reset();
			_line1 = stream1.readLine();
			findPair(false);
			return;
		}

		String[] tmp1 = _line1.split(";"), tmp2 = _line2.split(";");
		String rvl1 = tmp1[0], rvl2 = tmp2[0], qid1 = tmp1[1], qid2 = tmp2[1];
		if (new Integer(qid1) < new Integer(qid2)) {
			stream2.reset();
			_line1 = stream1.readLine();
			findPair(false);
		} else if (new Integer(qid1) > new Integer(qid2)) {
			findPair(true);
		} else {
			if (flag)
				stream2.mark(maxBufSize);
			if (rvl1.equals(rvl2))
				findPair(false);
			else {

				return;
			}
		}
	}

	/**
	 * Return the next token's double value in the current line
	 * 
	 * @return the next double value
	 */
	private double nextToken() throws java.io.IOException {
		return this.nextToken(null);
	}

	/**
	 * Return the next token's double value in the current line; tokens are
	 * separated by the characters contained in delim
	 * 
	 * @return the next double value
	 * @param delim
	 *            String containing the delimitators characters
	 */
	private double nextToken(String delim) throws java.io.IOException {
		double v;
		String nt = null;

		if (tokenizer == null)
			nextLine();
		if (delim != null)
			nt = tokenizer.nextToken(delim);
		else
			nt = tokenizer.nextToken();

		if (m_decimalPoint != '.')
			nt = nt.replace(m_decimalPoint, '.');
		try {
			v = Double.valueOf(nt).floatValue();
		} catch (NumberFormatException nfe) {
			log.warn("Warning: Not numeric value at row " + getLineno() + ": <"
					+ nt + ">");
			v = 0;
		}
		return v;
	}

	/**
	 * Go to the last marked position. Begin of input stream if no mark
	 * detected.
	 */
	public void resetInput() throws java.io.IOException {
		stream1.reset();
		tokenizer = null;
	}

	public void setDecimalPoint(char dp) {
		m_decimalPoint = dp;
	}

	public char getDecimalPoint() {
		return m_decimalPoint;
	}
}

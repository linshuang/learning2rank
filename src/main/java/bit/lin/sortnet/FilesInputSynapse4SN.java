package bit.lin.sortnet;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.TreeSet;

import org.joone.engine.FullSynapse;
import org.joone.engine.LinearLayer;
import org.joone.engine.Monitor;
import org.joone.engine.NetErrorManager;
import org.joone.engine.SigmoidLayer;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.exception.JooneRuntimeException;
import org.joone.io.FileInputSynapse;
import org.joone.io.StreamInputSynapse;
import org.joone.io.StreamInputTokenizer;
import org.joone.log.ILogger;
import org.joone.log.LoggerFactory;
import org.joone.net.NetCheck;
import org.joone.net.NeuralNet;

/**
 * this class is modified from $FileInputSynapse. and is used to receive many
 * files as input for pairwise learning.
 */
public class FilesInputSynapse4SN extends StreamInputSynapse {
	private static final long serialVersionUID = -5762261272795937845L;

	/** The logger used to log errors and warnings. */
	private static final ILogger log = LoggerFactory
			.getLogger(FileInputSynapse.class);

	/** The name of the file to extract information from. */
	private String fileName1 = "";
	private String fileName2 = "";
	private transient File inputFile1;
	private transient File inputFile2;

	public FilesInputSynapse4SN() {
		super();
	}

	private void readObject(ObjectInputStream in) throws IOException,
			ClassNotFoundException {
		super.readObjectBase(in);
		if (in.getClass().getName().indexOf("xstream") == -1) {
			fileName1 = (String) in.readObject();
		}
		if ((fileName1 != null) && (fileName1.length() > 0))
			inputFile1 = new File(fileName1);
	}

	private void writeObject(ObjectOutputStream out) throws IOException {
		super.writeObjectBase(out);
		if (out.getClass().getName().indexOf("xstream") == -1) {
			out.writeObject(fileName1);
		}
	}

	protected void initInputStream() throws JooneRuntimeException {
		if ((fileName1 != null) && (!fileName1.equals(new String("")))
				&& (fileName2 != null) && (!fileName2.equals(new String("")))) {
			try {
				inputFile1 = new File(fileName1);
				FileInputStream fis1 = new FileInputStream(inputFile1);
				inputFile2 = new File(fileName2);
				FileInputStream fis2 = new FileInputStream(inputFile2);
				StreamInputTokenizer4SN sit;
				if (getMaxBufSize() > 0) {
					sit = new StreamInputTokenizer4SN(new InputStreamReader(
							fis1), new InputStreamReader(fis2), getMaxBufSize());
				} else {
					sit = new StreamInputTokenizer4SN(new InputStreamReader(
							fis1), new InputStreamReader(fis2));
				}
				super.setTokens(sit);
			} catch (IOException ioe) {
				String error = "IOException in " + getName()
						+ ". Message is : ";
				log.warn(error + ioe.getMessage());
				if (getMonitor() != null)
					new NetErrorManager(getMonitor(), error + ioe.getMessage());
			}
		}
	}

	/**
	 * Returns a TreeSet of errors or problems regarding the setup of this
	 * synapse.
	 * 
	 * @return A TreeSet of errors or problems regarding the setup of this
	 *         synapse.
	 */
	public TreeSet check() {
		TreeSet checks = super.check();

		if (fileName1 == null || fileName1.trim().equals("")) {
			checks.add(new NetCheck(NetCheck.FATAL, "File Name not set.", this));
		} else {
			if (!inputFile1.exists()) {
				NetCheck error = new NetCheck(NetCheck.WARNING,
						"Input File doesn't exist.", this);
				if (getInputPatterns().isEmpty())
					error.setSeverity(NetCheck.FATAL);
				checks.add(error);
			}
		}

		return checks;
	}

	public File getInputFile() {
		return inputFile1;
	}

	public void setInputFile(File inputFile1, File inputFile2) {
		if (inputFile1 != null) {
			if (!fileName1.equals(inputFile1.getAbsolutePath())) {
				this.inputFile1 = inputFile1;
				fileName1 = inputFile1.getAbsolutePath();
				this.resetInput();
				super.setTokens(null);
			}
			if (!fileName2.equals(inputFile2.getAbsolutePath())) {
				this.inputFile2 = inputFile2;
				fileName2 = inputFile2.getAbsolutePath();
				this.resetInput();
				super.setTokens(null);
			}
		} else {
			this.inputFile1 = inputFile1;
			fileName1 = "";
			this.inputFile2 = inputFile2;
			fileName2 = "";
			this.resetInput();
			super.setTokens(null);
		}
	}

	public static void main(String[] args) {
		
	}
}

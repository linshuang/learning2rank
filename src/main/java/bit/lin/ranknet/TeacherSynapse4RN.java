package bit.lin.ranknet;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.joone.log.*;
import org.joone.engine.*;
import org.joone.engine.learning.AbstractTeacherSynapse;
import org.joone.engine.learning.TeacherSynapse;

/**
 * Final element of a neural network; it permits to calculate both the error of
 * the last training cycle and the vector containing the error pattern to apply
 * to the net to calculate the backprop algorithm.
 */
public class TeacherSynapse4RN extends TeacherSynapse {
	private static final long serialVersionUID = 1328436373640536321L;

	/**
	 * Logger
	 **/
	protected static final ILogger log = LoggerFactory
			.getLogger(TeacherSynapse4RN.class);

	/** The error being calculated for the current epoch. */
	protected transient double GlobalError = 0;

	public TeacherSynapse4RN() {
		super();
	}

	protected double calculateError(double aDesired, double anOutput,
			int anIndex) {
		// double myError = aDesired - anOutput;
		double myError = -aDesired + 1 / (1 + Math.exp(-anOutput))
				/ Math.log(10);

		// myError = Dn - Yn
		// myError^2 = (Dn - yn)^2
		// GlobalError += SUM[ SUM[ 1/2 (Dn - yn)^2]]
		// GlobalError += SUM[ 1/2 SUM[(Dn - yn)^2]]

		GlobalError += (myError * myError) / 2;
		return myError;
	}

	protected double calculateGlobalError() {
		double myError = GlobalError / getMonitor().getNumOfPatterns();
		if (getMonitor().isUseRMSE()) {
			myError = Math.sqrt(myError);
		}
		GlobalError = 0;
		return myError;
	}

	boolean f1 = true;

	public void fwdPut(Pattern pattern) {
		int step = pattern.getCount();

		if (!getMonitor().isSingleThreadMode()) {
			if ((getMonitor() == null) || (!isEnabled())) {
				if (step == -1) {
					stopTheNet();
				}
				return;
			}
		}

		if (isEnabled()) {
			synchronized (getFwdLock()) {
				while (items > 0) {
					try {
						fwdLock.wait();
					} catch (InterruptedException e) {
						reset();
						fwdLock.notify();
						return;
					} // End of catch
				}
				if (f1) {
					m_pattern = pattern;
					count = m_pattern.getCount();
					inps = (double[]) m_pattern.getArray();
					// forward(inps);
					++items;
					f1 = false;
				} else {
					merge(pattern);
					count = m_pattern.getCount();
					inps = (double[]) m_pattern.getArray();
					forward(inps);
					++items;
					f1 = true;
				}
				fwdLock.notify();
			}
		}

		if (step != -1) {
			if (!getMonitor().isLearningCicle(step)) {
				items = 0;
			}
		} else {
			items = 0;
		}

		if (pattern.getCount() == -1) {
			// reset error
			GlobalError = 0;
		}
	}

	private void merge(Pattern pattern) {

		List a = Arrays.asList(m_pattern.getArray());
		List b = Arrays.asList(pattern.getArray());
		a.addAll(b);
		double[] array = new double[a.size()];
		for (int i = 0; i < a.size(); i++) {
			array[i] = Double.parseDouble(a.get(i).toString());
		}
		m_pattern.setValues(array);

	}

	protected void forward(double[] pattern) {
		Pattern pattDesired;
		double[] pDesired;
		double myGlobalError; // error at the end of an epoch

		if ((m_pattern.getCount() == 1) || (m_pattern.getCount() == -1)) {
			// new epoch / end of previous epoch
			try {
				desired.gotoFirstLine();
				if ((!isFirstTime())
						&& (getSeenPatterns() == getMonitor()
								.getNumOfPatterns())) {
					myGlobalError = calculateGlobalError();
					pushError(myGlobalError, getMonitor().getTotCicles()
							- getMonitor().getCurrentCicle());
					getMonitor().setGlobalError(myGlobalError);
					epochFinished();
					setSeenPatterns(0);
				}
			} catch (IOException ioe) {
				new NetErrorManager(getMonitor(),
						"TeacherSynapse: IOException while forwarding the influx. Message is : "
								+ ioe.getMessage());
				return;
			}
		}
		if (m_pattern.getCount() == -1) {
			if (!getMonitor().isSingleThreadMode()) {
				stopTheNet();
			} else {
				pushError(0.0, -1);
			}
			return;
		}
		setFirstTime(false);
		outs = new double[pattern.length];
		pattDesired = desired.fwdGet();
		if (m_pattern.getCount() != pattDesired.getCount()) {
			try {
				desired.gotoLine(m_pattern.getCount());
				pattDesired = desired.fwdGet();
				if (m_pattern.getCount() != pattDesired.getCount()) {
					new NetErrorManager(getMonitor(),
							"TeacherSynapse: No matching patterns - input#"
									+ m_pattern.getCount() + " desired#"
									+ pattDesired.getCount());
					return;
				}
			} catch (IOException ioe) {
				new NetErrorManager(getMonitor(),
						"TeacherSynapse: IOException while forwarding the influx. Message is : "
								+ ioe.getMessage());
				return;
			}
		}

		// The error calculation starts from the preLearning+1 pattern
		if (getMonitor().getPreLearning() < m_pattern.getCount()) {
			pDesired = pattDesired.getArray();
			if (pDesired != null) {
				// if (pDesired.length != outs.length) {
				// // if the desired output differs in size, we will back
				// // propagate
				// // an pattern of the same size as the desired output so the
				// // output
				// // layer will adjust its size. The error pattern will
				// // contain zero
				// // values so no learning takes place during this backward
				// // pass.
				// log.warn("Size output pattern mismatches size desired pattern."
				// +
				// " Zero-valued desired pattern sized error pattern will be backpropagated.");
				// outs = new double[pDesired.length];
				// } else {
				// constructErrorPattern(pDesired, pattern);
				// }
				constructErrorPattern(pDesired, pattern);
			}
		}
		incSeenPatterns();
	}

	protected void constructErrorPattern(double[] aDesired, double[] anOutput) {
		int n = anOutput.length;
		for (int x = 0; x < aDesired.length; ++x) {
			outs[x] = calculateError(aDesired[x], anOutput[x], x);
			outs[n / 2 + x] = calculateError(aDesired[x], anOutput[n / 2 + x],
					x);
		}
		/**
		 * For debuging purpose to view the desired output String myText =
		 * "Desired: "; for (int x = 0; x < aDesired.length; ++x) { myText +=
		 * aDesired[x] + " "; } System.out.println(myText); end debug
		 */
	}

	public Pattern revGet() {
		if (!isEnabled())
			return null;
		synchronized (getFwdLock()) {
			if ((notFirstTime) || (!isLoopBack())) {
				while (items == 0) {
					try {
						fwdLock.wait();
					} catch (InterruptedException e) {
						// log.warn ( "wait () was interrupted");
						// e.printStackTrace();
						reset();
						fwdLock.notify();
						return null;
					}
				}
				--items;
				m_pattern.setArray(outs);
				if (isLoopBack())
					// To avoid sinc problems
					m_pattern.setCount(0);
				fwdLock.notify();
				return m_pattern;
			} else {
				items = bitems = count = 0;
				notFirstTime = true;
				fwdLock.notify();
				return null;
			}
		}
	}
}
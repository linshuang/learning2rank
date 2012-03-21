package bit.lin.sortnet;

import org.joone.log.*;
import org.joone.engine.*;
import org.joone.engine.learning.AbstractTeacherSynapse;
import org.joone.engine.learning.TeacherSynapse;

/**
 * Final element of a neural network; it permits to calculate both the error of
 * the last training cycle and the vector containing the error pattern to apply
 * to the net to calculate the backprop algorithm.
 */
public class TeacherSynapse4SN extends TeacherSynapse {
	private static final long serialVersionUID = 1328436373640536321L;

	/**
	 * Logger
	 **/
	protected static final ILogger log = LoggerFactory
			.getLogger(TeacherSynapse4SN.class);

	/** The error being calculated for the current epoch. */
	protected transient double GlobalError = 0;

	public TeacherSynapse4SN() {
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

	public void fwdPut(Pattern pattern) {
		super.fwdPut(pattern);

		if (pattern.getCount() == -1) {
			// reset error
			GlobalError = 0;
		}
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
                        //                    log.warn ( "wait () was interrupted");
                        //e.printStackTrace();
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
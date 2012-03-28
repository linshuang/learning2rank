package bit.lin.pairwise.ranknet;

import org.joone.engine.Learner;
import org.joone.engine.OutputPatternListener;
import org.joone.engine.Pattern;
import org.joone.engine.SigmoidLayer;
import org.joone.exception.JooneRuntimeException;
import org.joone.log.*;

/**
 * The output of a sigmoid layer neuron is the sum of the weighted input values,
 * applied to a sigmoid function. This function is expressed mathematically as:
 * y = 1 / (1 + e^-x) This has the effect of smoothly limiting the output within
 * the range 0 and 1
 * 
 * @see SimpleLayer parent
 * @see Layer parent
 * @see NeuralLayer implemented interface
 */
public class SigmoidLayer4RN extends SigmoidLayer {
	private static final long serialVersionUID = -966019257231874479L;

	private static final ILogger log = LoggerFactory
			.getLogger(SigmoidLayer4RN.class);

	/**
	 * The constructor
	 */
	public SigmoidLayer4RN() {
		super();
		learnable = true;
	}

	/**
	 * The constructor
	 * 
	 * @param ElemName
	 *            The name of the Layer
	 */
	public SigmoidLayer4RN(java.lang.String ElemName) {
		this();
		this.setLayerName(ElemName);
	}

	protected transient double[] outs0;

	public void backward(double[] pattern) throws JooneRuntimeException {
		super.backward(pattern);
		double dw, absv;
		int x;
		int n = getRows();
		double[] tmp = new double[n];
		for (x = 0; x < n; ++x) {
			gradientOuts[n / 2 + x] = pattern[n / 2 + x]
					* (outs[x] * (1 - outs[x]) + getFlatSpotConstant());
			gradientOuts[x] = pattern[x]
					* (outs0[x] * (1 - outs0[x]) + getFlatSpotConstant());
			tmp[x] = gradientOuts[n / 2 + x]- gradientOuts[x];
		}
		myLearner.requestBiasUpdate(tmp);
	}

	/**
	 * This method accepts an array of values in input and forwards it according
	 * to the Sigmoid propagation pattern.
	 * 
	 * @param pattern
	 * @see NeuralLayer#forward (double[])
	 * @throws JooneRuntimeException
	 *             This <code>Exception </code> is a wrapper Exception when an
	 *             Exception is thrown while doing the maths.
	 * */
	public void forward(double[] pattern) throws JooneRuntimeException {
		int x = 0;
		double in;
		int n = getRows();
		try {
			outs0 = outs.clone();
			for (x = 0; x < n; ++x) {
				in = pattern[x] + bias.value[x][0];
				outs[x] = 1 / (1 + Math.exp(-in));
			}
		} catch (Exception aioobe) {
			String msg;
			log.error(msg = "Exception thrown while processing the element "
					+ x + " of the array. Value is : " + pattern[x]
					+ " Exception thrown is " + aioobe.getClass().getName()
					+ ". Message is " + aioobe.getMessage());
			throw new JooneRuntimeException(msg, aioobe);
			// aioobe.printStackTrace();
		}
	}

	/**
	 * This method serves to a single backward step when the Layer is called
	 * from an external thread
	 */
	public void revRun(Pattern pattIn) {
		Pattern patt = new Pattern();
		gradientInps = new double[getDimension()];
		running = true;
		if (pattIn == null) {
			fireRevGet();
		} else {
			gradientInps = pattIn.getArray();
		}
		if (running) {
			backward(gradientInps);
			patt.setArray(gradientOuts);
			patt.setOutArray(outs);
			patt.setCount(step);
			fireRevPut(patt);
		}
		running = false;
	}

	/**
	 * Calls all the revGet methods on the output synapses to get the error
	 * gradients
	 */
	protected void fireRevGet() {
		if (outputPatternListeners == null) {
			return;
		}

		double[] patt;
		Pattern tPatt;
		int currentSize = outputPatternListeners.size();
		OutputPatternListener tempListener = null;
		for (int index = 0; (index < currentSize) && running; index++) {
			tempListener = (OutputPatternListener) outputPatternListeners
					.elementAt(index);
			if (tempListener != null) {
				tPatt = tempListener.revGet();
				if (tPatt != null) {
					patt = tPatt.getArray();
					// if (patt.length != gradientInps.length) {
					// adjustSizeToRevPattern(patt);
					// }

					// Sum the received error gradient pattern into outs.
					sumBackInput(patt);
				}
			}
		}
	}
}
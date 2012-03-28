package bit.lin.pairwise.sortnet;

import java.util.ArrayList;
import java.util.List;

import org.joone.engine.AbstractLearner;
import org.joone.engine.ExtendableLearner;
import org.joone.engine.RpropLearner;
import org.joone.engine.extenders.BatchModeExtender;
import org.joone.engine.extenders.DeltaRuleExtender;
import org.joone.engine.extenders.GradientExtender;
import org.joone.engine.extenders.MomentumExtender;
import org.joone.engine.extenders.OnlineModeExtender;
import org.joone.engine.extenders.RpropExtender;
import org.joone.engine.extenders.SimulatedAnnealingExtender;
import org.joone.engine.extenders.UpdateWeightExtender;

public class SortNetLearner extends ExtendableLearner {
	private static final long serialVersionUID = -2398703813090534579L;

	class SymmetricOnlineModeExtender extends UpdateWeightExtender {

		/** Creates a new instance of OnlineExtender */
		public SymmetricOnlineModeExtender() {
		}

		public void postBiasUpdate(double[] currentGradientOuts) {
		}

		public void postWeightUpdate(double[] currentPattern,
				double[] currentInps) {
		}

		public void preBiasUpdate(double[] currentGradientOuts) {
		}

		public void preWeightUpdate(double[] currentPattern,
				double[] currentInps) {
		}

		public void updateBias(int j, double aDelta) {
			getLearner().getLayer().getBias().delta[j][0] = aDelta;
			getLearner().getLayer().getBias().value[j][0] += aDelta;
			// n行
			int n = getLearner().getLayer().getBias().delta.length;
			if (j > (n - 1) / 2) {
				double tmp1 = (getLearner().getLayer().getBias().delta[j][0] + getLearner()
						.getLayer().getBias().delta[n - j - 1][0]) / 2;
				double tmp2 = (getLearner().getLayer().getBias().value[j][0] + getLearner()
						.getLayer().getBias().value[n - j - 1][0]) / 2;
				getLearner().getLayer().getBias().delta[j][0] = tmp1;
				getLearner().getLayer().getBias().delta[n - j - 1][0] = tmp1;

				getLearner().getLayer().getBias().value[j][0] = tmp2;
				getLearner().getLayer().getBias().value[n - j - 1][0] = tmp2;
			}
		}

		public void updateWeight(int j, int k, double aDelta) {
			getLearner().getSynapse().getWeights().delta[j][k] = aDelta;
			getLearner().getSynapse().getWeights().value[j][k] += aDelta;
			// n输入 行 m列
			int n = getLearner().getSynapse().getWeights().delta.length;
			int m = getLearner().getSynapse().getWeights().delta[0].length;
			if (j > (n - 1) / 2) {
				// n=4 -n/2
				if (n != 272) {
					double tmp1 = (getLearner().getSynapse().getWeights().delta[j][k] + getLearner()
							.getSynapse().getWeights().delta[n - j - 1][m - k
							- 1]) / 2;
					double tmp2 = (getLearner().getSynapse().getWeights().value[j][k] + getLearner()
							.getSynapse().getWeights().value[n - j - 1][m - k
							- 1]) / 2;
					getLearner().getSynapse().getWeights().delta[j][k] = tmp1;
					getLearner().getSynapse().getWeights().delta[n - j - 1][m
							- k - 1] = tmp1;
					getLearner().getSynapse().getWeights().value[j][k] = tmp2;
					getLearner().getSynapse().getWeights().value[n - j - 1][m
							- k - 1] = tmp2;
				} else {
					double tmp1 = (getLearner().getSynapse().getWeights().delta[j][k] + getLearner()
							.getSynapse().getWeights().delta[j - n / 2][m - k
							- 1]) / 2;
					double tmp2 = (getLearner().getSynapse().getWeights().value[j][k] + getLearner()
							.getSynapse().getWeights().value[j - n / 2][m - k
							- 1]) / 2;
					getLearner().getSynapse().getWeights().delta[j][k] = tmp1;
					getLearner().getSynapse().getWeights().delta[j - n / 2][m
							- k - 1] = tmp1;
					getLearner().getSynapse().getWeights().value[j][k] = tmp2;
					getLearner().getSynapse().getWeights().value[j - n / 2][m
							- k - 1] = tmp2;
				}
			}
		}

		public boolean storeWeightsBiases() {
			return true; // we will always store the weights / biases in the
							// online mode
		}

	}

	public SortNetLearner() {
		setUpdateWeightExtender(new OnlineModeExtender());
//		setUpdateWeightExtender(new BatchModeExtender());
		// please be careful of the order of extenders...
		addDeltaRuleExtender(new MomentumExtender());
		// addDeltaRuleExtender(new RpropExtender());
//		addDeltaRuleExtender(new SimulatedAnnealingExtender());
	}

	public static void main(String[] args) {
		RpropLearner p;
		double[][] a = new double[5][7];
		System.out.println(a.length);
		System.out.println(a[0].length);
		System.out.println(a.length / a[0].length);
	}
}

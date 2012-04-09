package bit.lin.pairwise.myrank;

import java.io.Serializable;

/**
 * this class is used to classify pair preference interval from -4 to 4. we try
 * to split this problem into 36 sub-problem, each of which is targeted to
 * classify the preference within two classes.
 */
public class PrfIntervalClassifier implements Serializable {
	private static final long serialVersionUID = 7009141929191976645L;

	/**
	 * index 0 with <-4,-3>; index 1 with <-4,-2>..<-3,-2>..<-2,-1>....index 35
	 * with <3,4>
	 */
	private SubClassifier[] _subClassifiers = SimpleNNArrayFactory
			.get36NNs(this);

	/**
	 * @param i
	 *            the preference of first pair
	 * @param j
	 *            the preference of second pair
	 */
	private SubClassifier getSubClassifier(int m, int n) {
		int i;
		int j;
		if (m < n) {
			i = m + 4;
			j = n + 4;
		} else {
			j = m + 4;
			i = n + 4;
		}
		int result = 0;
		int tmp1 = 8;
		int tmp2 = 0;
		for (; i > 0; i--) {
			result += tmp1;
			tmp1--;
			tmp2++;
		}
		result += (j - tmp2);
		return _subClassifiers[result - 1];
	}

	// /**
	// * @param i
	// * the preference of first pair
	// * @param j
	// * the preference of second pair
	// */
	// private SubClassifier[] getSubClassifiers(int m) {
	// for(int i =m;)
	// }
	/**
	 * @param index
	 *            the index of subclassifier
	 * @param flag
	 *            when it's true, preference1 is more trustful than preference2
	 * @return the voted preference
	 */
	private int getVotedPrf(int index, boolean flag) {
		int tmp1 = 8;
		while (index > 0) {
			if (index - tmp1 < 0)
				break;
			index -= tmp1;
			tmp1--;
		}
		// System.out.println((4 - tmp1) + ":" + (5 - tmp1 + index));
		return flag ? 4 - tmp1 : 5 - tmp1 + index;
	}

	public SubClassifier[] getSubClassifiers() {
		return _subClassifiers;
	}

	public int getPrf(String doc1, String doc2) {
		int[] result = new int[9];
		for (int i = 0; i < _subClassifiers.length; i++) {
			int votedP = getVotedPrf(i, _subClassifiers[i].classify(doc1, doc2)) + 4;
			result[votedP] += 1;
		}

		int idx = 0;
		int tmp = 0;
		for (int i = 0; i < result.length; i++) {
			if (result[i] > tmp) {
				tmp = result[i];
				idx = i;
			}
		}
		return idx - 4;
	}

	public boolean isStarted;

	/**
	 * 整体不在跑：首先必须不是启动了的，然后必须是没有子nn在跑的。 出於效率的目的，所以逻辑比较复杂。。。
	 */
	public boolean isTraing() {
		if (isStarted)
			return true;
		boolean flag = false;
		for (SubClassifier s : _subClassifiers) {
			flag = flag | s.isRunning();
			if (flag)
				return true;
		}
		return false;
	}

	public static void main(String[] args) {

	}
}

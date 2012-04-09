package bit.lin.pairwise.myrank;

public class SimpleNNArrayFactory {
	public static SimpleNN[] get36NNs(PrfIntervalClassifier pic) {
		SimpleNN[] sNNs = new SimpleNN[36];
		for (int i = 0; i < 36; i++) {
			int m = getPrf(i, true), n = getPrf(i, false);
			String sb = String.format("IDX: %d - PrfA: %d - PrfB: %d", i, m, n);

			sNNs[i] = new SimpleNN(sb, pic);
			sNNs[i]._pId[0] = m;
			sNNs[i]._pId[1] = n;
		}
		return sNNs;
	}

	private static int getPrf(int index, boolean flag) {
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
}

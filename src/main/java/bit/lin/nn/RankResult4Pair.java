package bit.lin.nn;

import java.util.ArrayList;

public class RankResult4Pair {

	private int _q = -1;

	public int getQ() {
		return _q;
	}

	private ArrayList<Double> _score = new ArrayList<Double>();

	public ArrayList<Double> getScore() {
		return this._score;
	}

	private ArrayList<Double[]> _truth = new ArrayList<Double[]>();

	public ArrayList<Double[]> getTruth() {
		return this._truth;
	}

	public String getTruthString() {
		String str = "";
		for (int i = 0; i < _truth.size(); i++) {
			if (_score.get(i) > 0.5)
				str += (_truth.get(i)[0] + " ");
			else
				str += (_truth.get(i)[1] + " ");
		}
		return str.trim() + "\n";
	}

	public void addS2T(double s, Double[] t) {
		_score.add(s);
		_truth.add(t);
	}

	public void sort() {
//		for (int i = 0; i < _score.size(); i++) {
//			for (int j = i + 1; j < _score.size(); j++) {
//				if (_score.get(j) > _score.get(i)) {
//					double tmpS = _score.get(j);
//					double tmpT = _truth.get(j);
//					_score.set(j, _score.get(i));
//					_truth.set(j, _truth.get(i));
//					_score.set(i, tmpS);
//					_truth.set(i, tmpT);
//				}
//			}
//		}
	}

	public void reset(String q) {
		_score = new ArrayList<Double>();
		_truth = new ArrayList<Double[]>();
		_q = new Integer(q);
	}
}

package bit.lin.nn;

import java.util.ArrayList;

public class RankResult {

	private int _q = -1;

	public int getQ() {
		return _q;
	}

	private ArrayList<Double> _score = new ArrayList<Double>();

	public ArrayList<Double> getScore() {
		return this._score;
	}

	private ArrayList<Double> _truth = new ArrayList<Double>();

	public ArrayList<Double> getTruth() {
		return this._truth;
	}

	public String getTruthString() {
		String str = "";
		for (int i = 0; i < _truth.size(); i++) {
			str += (_truth.get(i) + " ");
		}
		return str.trim() + "\n";
	}

	public void addS2T(double s, String t) {
		_score.add(s);
		_truth.add(new Double(t));
	}

	public void sort() {
		for (int i = 0; i < _score.size(); i++) {
			for (int j = i + 1; j < _score.size(); j++) {
				if (_score.get(j) > _score.get(i)) {
					double tmpS = _score.get(j);
					double tmpT = _truth.get(j);
					_score.set(j, _score.get(i));
					_truth.set(j, _truth.get(i));
					_score.set(i, tmpS);
					_truth.set(i, tmpT);
				}
			}
		}
	}

	public void reset(String q) {
		_score = new ArrayList<Double>();
		_truth = new ArrayList<Double>();
		_q = new Integer(q);
	}
}

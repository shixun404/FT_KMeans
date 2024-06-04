
#define DI inline __device__

namespace my_test{
	#ifndef C_H
	#define C_H
	template <typename _Key, typename _Value>
	struct KeyValuePair {
		typedef _Key Key;      ///< Key data type
		typedef _Value Value;  ///< Value data type

		Key key;      ///< Item key
		Value value;  ///< Item value

		/// Constructor
		KeyValuePair() = default;

		/// Constructor
		inline KeyValuePair(Key const& key, Value const& value) : key(key), value(value) {}

		/// Inequality operator
		inline bool operator!=(const KeyValuePair& b)
		{
			return (value != b.value) || (key != b.key);
		}

		inline bool operator<(const KeyValuePair<_Key, _Value>& b) const
		{
			return (key < b.key) || ((key == b.key) && value < b.value);
		}

		inline bool operator>(const KeyValuePair<_Key, _Value>& b) const
		{
			return (key > b.key) || ((key == b.key) && value > b.value);
		}
	};

	template <typename AccType, typename Index, typename OutType>
	struct kvp_cg_min_reduce_op {
		typedef KeyValuePair<Index, AccType> KVP;

		__host__ __device__ kvp_cg_min_reduce_op() noexcept {};

		using AccTypeT = AccType;
		using IndexT   = Index;
		// functor signature.
		__host__ __device__ KVP operator()(KVP a, KVP b) const { return a.value < b.value ? a : b; }

		__host__ __device__ AccType operator()(AccType a, AccType b) const { return min(a, b); }

		__host__ __device__ bool isAmin(AccType a, AccType b) const { return a < b ? true : false; }
	};

	template <typename LabelT, typename DataT>
	struct MinAndDistanceReduceOpImpl {
		typedef KeyValuePair<LabelT, DataT> KVP;
		DI void operator()(LabelT rid, KVP* out, const KVP& other) const
		{
			if (other.value < out->value) {
			out->key   = other.key;
			out->value = other.value;
			}
		}

		DI void operator()(LabelT rid, DataT* out, const KVP& other) const
		{
			if (other.value < *out) { *out = other.value; }
		}

		DI void operator()(LabelT rid, DataT* out, const DataT& other) const
		{
			if (other < *out) { *out = other; }
		}

		DI void init(DataT* out, DataT maxVal) const { *out = maxVal; }
		DI void init(KVP* out, DataT maxVal) const { out->value = maxVal; }

		DI void init_key(DataT& out, LabelT idx) const { return; }
		DI void init_key(KVP& out, LabelT idx) const { out.key = idx; }

		DI DataT get_value(KVP& out) const
		{
			return out.value;
			;
		}
		DI DataT get_value(DataT& out) const { return out; }
	};

	template <typename LabelT, typename DataT>
	struct KVPMinReduceImpl {
		typedef KeyValuePair<LabelT, DataT> KVP;
		DI KVP operator()(LabelT rit, const KVP& a, const KVP& b) { return b.value < a.value ? b : a; }
		DI KVP operator()(const KVP& a, const KVP& b) { return b.value < a.value ? b : a; }

	}; 
	#endif // C_H
}

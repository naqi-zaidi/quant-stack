// Python bindings for the order book (Project 2, step 5): the extension module orderbook._lob.
//
// What Python gets
//   OrderBook, ArrayOrderBook       a book Python owns: submit, cancel, modify, every read-only
//                                   view, and the event stream (take_events, digest)
//   OrderBookView, ArrayOrderBookView   the read-only views alone (what Replay.book returns)
//   Trade, LevelUpdate, ExecReport  the three event types        } named tuples
//   Result, LevelView, OrderView    what calls and views return  } made in C++
//   lobster.Replay, ArrayReplay     LOBSTER replay, one event at a time or a whole day in C++
//   lobster.DayReader, Message, Snapshot, ReplayStats, ...
//   analytics.SpreadDistribution, DepthProfile, OrderFlowImbalance, QueuePositions
//                                   listeners attached to a book or replay with attach()
//
// Prices cross the boundary as integer ticks (Python int), never floats. Events stay in a C++
// vector until take_events() copies them out, and the FNV-1a digest (digest.hpp) folds in
// every event whether it is recorded or not, so Python can check it received the exact stream.
// Objects are not thread-safe: use one book from one thread.
//
// The enums are real Python IntEnum classes, made here with the enum module, with their own
// type casters: Python -> C++ is a type check plus PyLong_AsLong (an IntEnum member is an int),
// C++ -> Python hands back a cached member. pybind11's generic native_enum path reads the
// member's .value property instead, which made every enum argument cost ~300 ns.
//
// The six value types are PyStructSequence named tuples filled directly with the C API:
// immutable, picklable, usable in `match`, and ~4x cheaper to create than a pybind11 class
// instance (which heap-copies the value and registers it), so a call returning a Result costs
// ~100 ns less.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <tuple>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "lob/analytics.hpp"
#include "lob/digest.hpp"
#include "lob/events.hpp"
#include "lob/lobster.hpp"
#include "lob/order_book.hpp"
#include "lob/text.hpp"
#include "lob/types.hpp"

namespace py = pybind11;

namespace lob::python {

// One Python IntEnum class per C++ enum. Strong references, kept for the life of the process.
template <typename E>
struct PyEnum {
  static inline PyObject* type = nullptr;
  static inline std::vector<PyObject*> members;  // indexed by value; null where none
};

template <typename E>
bool load_enum(py::handle src, E& out) {
  PyObject* type = PyEnum<E>::type;
  if (type == nullptr || !PyObject_TypeCheck(src.ptr(), reinterpret_cast<PyTypeObject*>(type))) {
    return false;
  }
  const long v = PyLong_AsLong(src.ptr());
  if (v == -1 && PyErr_Occurred() != nullptr) {
    PyErr_Clear();
    return false;
  }
  out = static_cast<E>(v);
  return true;
}

template <typename E>
py::handle cast_enum(E e) {
  const auto i = static_cast<std::size_t>(e);
  const auto& members = PyEnum<E>::members;
  if (i >= members.size() || members[i] == nullptr) {
    PyErr_SetString(PyExc_ValueError, "C++ enum value has no Python member");
    return {};
  }
  return py::handle(members[i]).inc_ref();
}

// Creates the IntEnum `name` in `m` with the given members and registers it for the casters.
template <typename E>
void def_enum(py::module_& m, const char* name, const char* doc,
              std::initializer_list<std::pair<const char*, E>> values) {
  py::list items;
  std::size_t top = 0;
  for (const auto& [n, v] : values) {
    items.append(py::make_tuple(n, static_cast<int>(v)));
    top = std::max(top, static_cast<std::size_t>(v));
  }
  py::object cls = py::module_::import("enum").attr("IntEnum")(
      name, items, py::arg("module") = m.attr("__name__"), py::arg("qualname") = name);
  cls.attr("__doc__") = doc;
  m.attr(name) = cls;
  PyEnum<E>::members.assign(top + 1, nullptr);
  for (const auto& [n, v] : values) {
    PyEnum<E>::members[static_cast<std::size_t>(v)] = py::object(cls.attr(n)).release().ptr();
  }
  PyEnum<E>::type = cls.release().ptr();
}

// ---- named tuples ----

template <typename T>
struct PySeq {
  static inline PyTypeObject* type = nullptr;
};

// Creates the struct sequence `module.name` with these fields and adds it to `m`.
template <typename T>
void def_seq(py::module_& m, const char* qualified_name, const char* doc,
             std::initializer_list<std::pair<const char*, const char*>> fields) {
  // The descriptors must outlive the type: leak them, once per type.
  auto* f = new PyStructSequence_Field[fields.size() + 1];
  std::size_t i = 0;
  for (const auto& [name, fdoc] : fields) f[i++] = {name, fdoc};
  f[i] = {nullptr, nullptr};
  auto* desc = new PyStructSequence_Desc{qualified_name, doc, f, static_cast<int>(fields.size())};
  PyTypeObject* type = PyStructSequence_NewType(desc);
  if (type == nullptr) throw py::error_already_set();
  PySeq<T>::type = type;  // keeps the new reference for the life of the process
  const std::string q(qualified_name);
  m.attr(q.substr(q.rfind('.') + 1).c_str()) = py::handle(reinterpret_cast<PyObject*>(type));
}

PyObject* to_py_int(std::int64_t v) { return PyLong_FromLongLong(v); }
PyObject* to_py_uint(std::uint64_t v) { return PyLong_FromUnsignedLongLong(v); }
template <typename E>
PyObject* to_py_enum(E e) { return cast_enum(e).ptr(); }

// Fills a new struct sequence of T's type from already-created items (new references).
template <typename T, std::size_t N>
py::handle make_seq(const std::array<PyObject*, N>& items) {
  PyObject* obj = PySeq<T>::type ? PyStructSequence_New(PySeq<T>::type) : nullptr;
  bool ok = obj != nullptr;
  for (PyObject* it : items) ok = ok && it != nullptr;
  if (!ok) {
    for (PyObject* it : items) Py_XDECREF(it);
    Py_XDECREF(obj);
    if (PyErr_Occurred() == nullptr) PyErr_SetString(PyExc_RuntimeError, "orderbook type missing");
    return {};
  }
  for (std::size_t k = 0; k < N; ++k) PyStructSequence_SetItem(obj, static_cast<Py_ssize_t>(k), items[k]);
  return obj;
}

py::handle to_py(const Trade& t) {
  return make_seq<Trade, 6>({to_py_uint(t.taker), to_py_uint(t.maker), to_py_enum(t.taker_side),
                             to_py_int(t.price.ticks), to_py_int(t.qty), to_py_uint(t.ts)});
}
py::handle to_py(const LevelUpdate& u) {
  return make_seq<LevelUpdate, 5>({to_py_enum(u.side), to_py_int(u.price.ticks), to_py_int(u.qty),
                                   to_py_uint(u.count), to_py_uint(u.ts)});
}
py::handle to_py(const ExecReport& r) {
  return make_seq<ExecReport, 8>({to_py_uint(r.id), to_py_enum(r.type), to_py_enum(r.reason),
                                  to_py_enum(r.side), to_py_int(r.price.ticks),
                                  to_py_int(r.last_qty), to_py_int(r.leaves), to_py_uint(r.ts)});
}
py::handle to_py(const Result& r) {
  return make_seq<Result, 5>({PyBool_FromLong(r.accepted ? 1 : 0), to_py_enum(r.reason),
                              to_py_int(r.filled), to_py_int(r.resting), to_py_int(r.cancelled)});
}
py::handle to_py(const LevelView& v) {
  return make_seq<LevelView, 3>({to_py_int(v.price.ticks), to_py_int(v.qty), to_py_uint(v.count)});
}
PyObject* to_py_price(const std::optional<Price>& p) {
  if (!p) Py_RETURN_NONE;
  return to_py_int(p->ticks);
}
py::handle to_py(const analytics::Quote& q) {
  return make_seq<analytics::Quote, 4>({to_py_price(q.bid), to_py_int(q.bid_qty),
                                        to_py_price(q.ask), to_py_int(q.ask_qty)});
}
py::handle to_py(const analytics::QueuePosition& p) {
  return make_seq<analytics::QueuePosition, 2>({to_py_uint(p.orders), to_py_int(p.qty)});
}
py::handle to_py(const OrderView& o) {
  return make_seq<OrderView, 6>({to_py_uint(o.id), to_py_enum(o.side), to_py_enum(o.type),
                                 to_py_int(o.price.ticks), to_py_int(o.qty), to_py_uint(o.ts)});
}

}  // namespace lob::python

namespace pybind11::detail {
#define QS_INT_ENUM_CASTER(T, PYNAME)                                                     \
  template <>                                                                             \
  struct type_caster<T> {                                                                 \
    PYBIND11_TYPE_CASTER(T, const_name(PYNAME));                                          \
    bool load(handle src, bool) { return ::lob::python::load_enum(src, value); }          \
    static handle cast(T src, return_value_policy, handle) {                              \
      return ::lob::python::cast_enum(src);                                               \
    }                                                                                     \
  };
QS_INT_ENUM_CASTER(lob::Side, "orderbook.Side")
QS_INT_ENUM_CASTER(lob::OrderType, "orderbook.OrderType")
QS_INT_ENUM_CASTER(lob::ExecType, "orderbook.ExecType")
QS_INT_ENUM_CASTER(lob::Reason, "orderbook.Reason")
QS_INT_ENUM_CASTER(lob::lobster::Event, "orderbook.lobster.Event")
QS_INT_ENUM_CASTER(lob::analytics::QueueStatus, "orderbook.analytics.QueueStatus")
#undef QS_INT_ENUM_CASTER

// Output-only: Python never passes these back in.
#define QS_SEQ_CASTER(T, PYNAME)                                                          \
  template <>                                                                             \
  struct type_caster<T> {                                                                 \
    PYBIND11_TYPE_CASTER(T, const_name(PYNAME));                                          \
    bool load(handle, bool) { return false; }                                             \
    static handle cast(const T& src, return_value_policy, handle) {                       \
      return ::lob::python::to_py(src);                                                   \
    }                                                                                     \
  };
QS_SEQ_CASTER(lob::Trade, "orderbook.Trade")
QS_SEQ_CASTER(lob::LevelUpdate, "orderbook.LevelUpdate")
QS_SEQ_CASTER(lob::ExecReport, "orderbook.ExecReport")
QS_SEQ_CASTER(lob::Result, "orderbook.Result")
QS_SEQ_CASTER(lob::LevelView, "orderbook.LevelView")
QS_SEQ_CASTER(lob::OrderView, "orderbook.OrderView")
QS_SEQ_CASTER(lob::analytics::Quote, "orderbook.analytics.Quote")
QS_SEQ_CASTER(lob::analytics::QueuePosition, "orderbook.analytics.QueuePosition")
#undef QS_SEQ_CASTER
}  // namespace pybind11::detail

namespace lob::python {
namespace {

using Event = std::variant<Trade, LevelUpdate, ExecReport>;

// The listener behind every book and replay Python owns. It counts every event, folds it into
// the digest unless hashing is off, and keeps it for take_events() unless recording is off.
class Sink final : public Listener {
 public:
  void on_trade(const Trade& t) override {
    ++count;
    if (hash) digest.on_trade(t);
    if (record) events.emplace_back(t);
    for (Listener* l : attached) l->on_trade(t);
  }
  void on_level_update(const LevelUpdate& u) override {
    ++count;
    if (hash) digest.on_level_update(u);
    if (record) events.emplace_back(u);
    for (Listener* l : attached) l->on_level_update(u);
  }
  void on_report(const ExecReport& r) override {
    ++count;
    if (hash) digest.on_report(r);
    if (record) events.emplace_back(r);
    for (Listener* l : attached) l->on_report(r);
  }
  // An analytic rebuilds its view of the book from events, so it must see them all.
  void attach(Listener& l) {
    if (count != 0) {
      throw std::runtime_error("attach() listeners before the first event: this one has seen " +
                               std::to_string(count));
    }
    attached.push_back(&l);
  }
  std::vector<Event> take() { return std::exchange(events, {}); }
  [[nodiscard]] std::optional<std::uint64_t> value() const {
    return hash ? std::optional<std::uint64_t>(digest.value()) : std::nullopt;
  }

  EventDigest digest;
  std::vector<Event> events;
  std::vector<Listener*> attached;
  std::size_t count{0};
  bool hash{true};
  bool record{true};
};

// A book that reports to its own Sink. The base is built first with no listener; the sink is
// attached once it exists.
template <typename Levels>
class OwnedBook final : public BasicOrderBook<Levels> {
 public:
  OwnedBook(std::int64_t tick_size, std::size_t expected_orders, bool record, bool hash)
      : BasicOrderBook<Levels>(nullptr, expected_orders, tick_size) {
    sink.record = record;
    sink.hash = hash;
    this->set_listener(&sink);
  }
  Sink sink;
};

// A LOBSTER replay that reports to its own Sink, and can run a whole day without Python.
template <typename Book>
class OwnedReplay final : public lobster::BasicReplay<Book> {
  using Base = lobster::BasicReplay<Book>;

 public:
  OwnedReplay(std::size_t levels, std::int64_t tick, bool record, bool hash)
      : Base(levels, tick), levels_(levels) {
    sink.record = record;
    sink.hash = hash;
    this->set_listener(&sink);
  }

  void seed(const lobster::Snapshot& s, Timestamp ts) {
    started_ = true;
    Base::seed(s, ts);
  }
  bool step(const lobster::Message& m, const lobster::Snapshot& after) {
    started_ = true;
    return Base::step(m, after);
  }

  // The first event seeds the book, every later one is stepped and compared: the same loop
  // as lob_replay. Runs with the GIL released.
  lobster::ReplayStats run(const std::string& message_path,
                           const std::optional<std::string>& orderbook_path) {
    if (started_) throw std::runtime_error("run() needs a fresh replay; this one has started");
    const std::string ob =
        orderbook_path ? *orderbook_path : lobster::describe_file(message_path).orderbook_path;
    lobster::DayReader rd(message_path, ob, levels_);
    lobster::Message m;
    lobster::Snapshot s;
    if (!rd.next(m, s)) throw std::runtime_error("empty LOBSTER file: " + message_path);
    seed(s, m.ns);
    while (rd.next(m, s)) step(m, s);
    return this->stats();
  }

  [[nodiscard]] std::size_t levels() const noexcept { return levels_; }
  Sink sink;

 private:
  std::size_t levels_;
  bool started_{false};
};

// A day's two files read in lockstep, as a Python iterator of (Message, Snapshot).
class PyDayReader {
 public:
  PyDayReader(const std::string& message_path, const std::optional<std::string>& orderbook_path,
              std::optional<std::size_t> levels)
      : reader_(message_path, resolve_path(message_path, orderbook_path),
                resolve_levels(message_path, levels)) {}

  std::pair<lobster::Message, lobster::Snapshot> next() {
    lobster::Message m;
    lobster::Snapshot s;
    if (!reader_.next(m, s)) throw py::stop_iteration();
    return {m, std::move(s)};
  }
  [[nodiscard]] std::size_t rows() const noexcept { return reader_.rows(); }

 private:
  static std::string resolve_path(const std::string& msg, const std::optional<std::string>& ob) {
    return ob ? *ob : lobster::describe_file(msg).orderbook_path;
  }
  static std::size_t resolve_levels(const std::string& msg, std::optional<std::size_t> levels) {
    return levels ? *levels : lobster::describe_file(msg).levels;
  }
  lobster::DayReader reader_;
};

using Levels = std::vector<std::pair<std::int64_t, Quantity>>;

Levels to_pairs(const std::vector<lobster::DepthLevel>& v) {
  Levels out;
  out.reserve(v.size());
  for (const auto& d : v) out.emplace_back(d.price.ticks, d.qty);
  return out;
}
std::vector<lobster::DepthLevel> from_pairs(const Levels& v) {
  std::vector<lobster::DepthLevel> out;
  out.reserve(v.size());
  for (const auto& [px, q] : v) out.push_back({Price{px}, q});
  return out;
}

// ---- books ----

template <typename Levels_>
void bind_book(py::module_& m, const char* view_name, const char* name, const char* what) {
  using View = BasicOrderBook<Levels_>;
  using Owned = OwnedBook<Levels_>;

  const std::string view_doc = std::string("Read-only views of a book with ") + what +
                               ". Returned by Replay.book; " + name +
                               " adds the calls that change the book.";
  py::class_<View>(m, view_name, view_doc.c_str())
      .def("best_bid",
           [](const View& b) -> std::optional<std::int64_t> {
             const auto p = b.best_bid();
             return p ? std::optional<std::int64_t>(p->ticks) : std::nullopt;
           },
           "Best bid price in ticks, or None if there are no bids.")
      .def("best_ask",
           [](const View& b) -> std::optional<std::int64_t> {
             const auto p = b.best_ask();
             return p ? std::optional<std::int64_t>(p->ticks) : std::nullopt;
           },
           "Best ask price in ticks, or None if there are no asks.")
      .def("best", &View::best, py::arg("side"), "Best level of one side, or None.")
      .def("depth", &View::depth, py::arg("side"), py::arg("max_levels") = std::size_t{10},
           "Up to max_levels levels of one side, best first.")
      .def("queue",
           [](const View& b, Side side, std::int64_t price) { return b.queue(side, Price{price}); },
           py::arg("side"), py::arg("price"),
           "Order ids resting at one price, in priority order (empty if no such level).")
      .def("orders", &View::orders, py::arg("side"),
           "Every resting order on one side in priority order: best price first, oldest first "
           "within a price.")
      .def("find", &View::find, py::arg("id"), "The resting order with this id, or None.")
      .def_property_readonly("order_count", &View::order_count, "Resting orders, both sides.")
      .def("level_count", &View::level_count, py::arg("side"), "Price levels on one side.")
      .def("resting_qty", &View::resting_qty, py::arg("side"), "Total resting size on one side.")
      .def_property_readonly("tick_size", &View::tick_size)
      .def("validate", &View::validate,
           "Full structural self-check; returns '' if the book is sound, else the first problem.")
      .def("to_text", [](const View& b) { return to_text(b); },
           "The book as a ladder: asks worst to best, a rule, bids best to worst.")
      .def("__len__", &View::order_count);

  const std::string doc = std::string("Price-time priority order book with ") + what +
                          ". Records every event it emits until take_events() collects them.";
  py::class_<Owned, View>(m, name, doc.c_str())
      .def(py::init<std::int64_t, std::size_t, bool, bool>(), py::kw_only(),
           py::arg("tick_size") = std::int64_t{1},
           py::arg("expected_orders") = std::size_t{1} << 16,
           py::arg("record_events") = true, py::arg("digest_events") = true,
           "Prices must be positive multiples of tick_size. expected_orders sizes the order pool "
           "and id index up front. digest_events=False skips the digest (~100 ns per event).")
      .def("submit",
           [](Owned& b, OrderId id, Side side, std::int64_t price, Quantity qty, OrderType type,
              Timestamp ts) { return b.submit(Order{id, side, type, Price{price}, qty, ts}); },
           py::arg("id"), py::arg("side"), py::arg("price"), py::arg("qty"),
           py::arg("type") = OrderType::Limit, py::arg("ts") = Timestamp{0},
           "New order of any type. The price of a Market order is ignored.")
      .def("cancel", [](Owned& b, OrderId id, Timestamp ts) { return b.cancel(id, ts); },
           py::arg("id"), py::arg("ts") = Timestamp{0}, "Removes a resting order.")
      .def("modify",
           [](Owned& b, OrderId id, std::int64_t price, Quantity qty, Timestamp ts) {
             return b.modify(id, Price{price}, qty, ts);
           },
           py::arg("id"), py::arg("price"), py::arg("qty"), py::arg("ts") = Timestamp{0},
           "New price and open quantity. A size cut at the same price keeps queue priority; "
           "anything else is a cancel-replace.")
      .def("take_events", [](Owned& b) { return b.sink.take(); },
           "Events recorded since the last call, in the order the book emitted them.")
      .def("attach", [](Owned& b, Listener& l) { b.sink.attach(l); }, py::arg("listener"),
           py::keep_alive<1, 2>(),
           "Sends every event to an orderbook.analytics listener as well. Attach before the "
           "first call; the book keeps the listener alive.")
      .def_property_readonly("digest", [](const Owned& b) { return b.sink.value(); },
                             "FNV-1a 64 digest of every event emitted so far (digest.hpp), or "
                             "None if the book was made with digest_events=False.")
      .def_property_readonly("event_count", [](const Owned& b) { return b.sink.count; },
                             "Events emitted so far, recorded or not.")
      .def_property(
          "record_events", [](const Owned& b) { return b.sink.record; },
          [](Owned& b, bool on) { b.sink.record = on; },
          "Keep events for take_events(). Off, only the digest and count advance.")
      .def("__repr__", [name](const Owned& b) {
        std::ostringstream os;
        os << name << "(tick_size=" << b.tick_size() << "): " << b.order_count() << " orders";
        return os.str();
      });
}

template <typename Book>
void bind_replay(py::module_& m, const char* name, const char* what) {
  using R = OwnedReplay<Book>;
  const std::string doc = std::string("LOBSTER replay into a book with ") + what +
                          ": rebuilds the day from its message file and compares the book with "
                          "LOBSTER's after every event.";
  py::class_<R>(m, name, doc.c_str())
      .def(py::init<std::size_t, std::int64_t, bool, bool>(), py::arg("levels"),
           py::arg("tick") = std::int64_t{1}, py::kw_only(), py::arg("record_events") = false,
           py::arg("digest_events") = true,
           "levels: depth of the LOBSTER files. tick: price grid (100 = one cent).")
      .def("seed", &R::seed, py::arg("snapshot"), py::arg("ts") = Timestamp{0},
           "Seeds the book from the snapshot after the day's first message.")
      .def("step", &R::step, py::arg("message"), py::arg("snapshot"),
           "Applies one event and compares every visible level; True if all matched.")
      .def("run", &R::run, py::arg("message_path"), py::arg("orderbook_path") = py::none(),
           py::call_guard<py::gil_scoped_release>(),
           "Replays a whole day in C++ with the GIL released and returns the stats. The "
           "orderbook file is found from the message file's name unless given.")
      .def_property_readonly("stats", [](const R& r) { return r.stats(); })
      .def_property_readonly("book", &R::book, py::return_value_policy::reference_internal,
                             "Read-only view of the replayed book.")
      .def_property_readonly("levels", &R::levels)
      .def("take_events", [](R& r) { return r.sink.take(); },
           "Book events recorded since the last call (record_events must be on).")
      .def("attach", [](R& r, Listener& l) { r.sink.attach(l); }, py::arg("listener"),
           py::keep_alive<1, 2>(),
           "Sends every book event to an orderbook.analytics listener as well, seeding "
           "included. Attach before seed() or run().")
      .def_property_readonly("digest", [](const R& r) { return r.sink.value(); },
                             "FNV-1a 64 digest of every book event so far (None if off).")
      .def_property_readonly("event_count", [](const R& r) { return r.sink.count; })
      .def_property(
          "record_events", [](const R& r) { return r.sink.record; },
          [](R& r, bool on) { r.sink.record = on; });
}

template <typename T>
py::array_t<T> column(std::size_t n) {
  return py::array_t<T>(static_cast<py::ssize_t>(n));
}

void bind_analytics(py::module_& am) {
  using namespace lob::analytics;
  def_enum<QueueStatus>(am, "QueueStatus", "How a tracked order's history sample came about.",
                        {{"Resting", QueueStatus::Resting},
                         {"Filled", QueueStatus::Filled},
                         {"Cancelled", QueueStatus::Cancelled}});
  def_seq<Quote>(am, "orderbook.analytics.Quote",
                 "Best bid and ask with their sizes; a missing side has price None and size 0.",
                 {{"bid", "ticks or None"}, {"bid_qty", "size"},
                  {"ask", "ticks or None"}, {"ask_qty", "size"}});
  def_seq<QueuePosition>(am, "orderbook.analytics.QueuePosition",
                         "Orders and open size ahead of an order in its queue.",
                         {{"orders", "orders ahead"}, {"qty", "their open size"}});
  am.attr("NO_BID") = std::numeric_limits<std::int64_t>::min();
  am.attr("NO_ASK") = std::numeric_limits<std::int64_t>::max();

  py::class_<Listener>(am, "Listener",
                       "Base of the analytics: attach one to a book or replay with attach().");

  py::class_<SpreadDistribution, Listener>(
      am, "SpreadDistribution", "Time-weighted distribution of the bid-ask spread, in ticks.")
      .def(py::init<std::int64_t>(), py::arg("tick_size") = std::int64_t{1})
      .def_property_readonly("spread", &SpreadDistribution::spread,
                             "Current spread in ticks, or None if a side is empty.")
      .def("histogram", &SpreadDistribution::histogram, py::arg("now"),
           "(spread in ticks, ns spent there) pairs, ascending, up to `now`.")
      .def("two_sided_ns", &SpreadDistribution::two_sided_ns, py::arg("now"))
      .def("one_sided_ns", &SpreadDistribution::one_sided_ns, py::arg("now"))
      .def("mean", &SpreadDistribution::mean, py::arg("now"),
           "Time-weighted mean spread in ticks over two-sided time.")
      .def("elapsed", &SpreadDistribution::elapsed, py::arg("now"))
      .def("reset", &SpreadDistribution::reset, py::arg("now"),
           "Restarts the statistics at `now`, keeping the rebuilt book.")
      .def_property_readonly("quote", &SpreadDistribution::quote)
      .def_property_readonly("tick_size", &SpreadDistribution::tick_size)
      .def_property_readonly("overflowed", &SpreadDistribution::overflowed);

  py::class_<DepthProfile, Listener>(
      am, "DepthProfile",
      "Time-weighted mean resting size at 0..ticks-1 ticks from each side's best price.")
      .def(py::init<std::int64_t, std::size_t>(), py::arg("tick_size") = std::int64_t{1},
           py::arg("ticks") = std::size_t{10})
      .def("current", &DepthProfile::current, py::arg("side"))
      .def("sums", &DepthProfile::sums, py::arg("side"), py::arg("now"),
           "Size x nanoseconds summed over time at each distance.")
      .def("observed_ns", &DepthProfile::observed_ns, py::arg("side"), py::arg("now"),
           "Time the side had at least one level.")
      .def("mean", &DepthProfile::mean, py::arg("side"), py::arg("now"))
      .def("elapsed", &DepthProfile::elapsed, py::arg("now"))
      .def("reset", &DepthProfile::reset, py::arg("now"))
      .def_property_readonly("quote", &DepthProfile::quote)
      .def_property_readonly("ticks", &DepthProfile::ticks)
      .def_property_readonly("tick_size", &DepthProfile::tick_size)
      .def_property_readonly("overflowed", &DepthProfile::overflowed);

  py::class_<OrderFlowImbalance, Listener>(
      am, "OrderFlowImbalance",
      "Cont-Kukanov-Stoikov order-flow imbalance: e_n at every change of the best quotes, the "
      "sum of the last `window`, and the running total.")
      .def(py::init<std::size_t, bool>(), py::arg("window") = std::size_t{100},
           py::arg("record") = false)
      .def_property_readonly("last", &OrderFlowImbalance::last)
      .def_property_readonly("window_sum", &OrderFlowImbalance::window_sum)
      .def_property_readonly("window", &OrderFlowImbalance::window)
      .def_property_readonly("window_count", &OrderFlowImbalance::window_count)
      .def_property_readonly("total", &OrderFlowImbalance::total)
      .def_property_readonly("updates", &OrderFlowImbalance::updates)
      .def_property_readonly("quote", &OrderFlowImbalance::quote)
      .def_property_readonly("imbalance", &OrderFlowImbalance::imbalance,
                             "(bid size - ask size) / (bid size + ask size) at the best "
                             "quotes, or None if a side is empty.")
      .def("reset", &OrderFlowImbalance::reset)
      .def(
          "records",
          [](const OrderFlowImbalance& o) {
            const auto& r = o.records();
            const std::size_t n = r.size();
            auto ts = column<std::uint64_t>(n);
            auto bid = column<std::int64_t>(n);
            auto bq = column<std::int64_t>(n);
            auto ask = column<std::int64_t>(n);
            auto aq = column<std::int64_t>(n);
            auto e = column<std::int64_t>(n);
            auto* pts = ts.mutable_data();
            auto* pb = bid.mutable_data();
            auto* pbq = bq.mutable_data();
            auto* pa = ask.mutable_data();
            auto* paq = aq.mutable_data();
            auto* pe = e.mutable_data();
            for (std::size_t i = 0; i < n; ++i) {
              pts[i] = r[i].ts;
              pb[i] = r[i].quote.bid ? r[i].quote.bid->ticks
                                     : std::numeric_limits<std::int64_t>::min();
              pbq[i] = r[i].quote.bid_qty;
              pa[i] = r[i].quote.ask ? r[i].quote.ask->ticks
                                     : std::numeric_limits<std::int64_t>::max();
              paq[i] = r[i].quote.ask_qty;
              pe[i] = r[i].e;
            }
            py::dict d;
            d["ts"] = ts;
            d["bid"] = bid;
            d["bid_qty"] = bq;
            d["ask"] = ask;
            d["ask_qty"] = aq;
            d["e"] = e;
            return d;
          },
          "Every quote update as numpy columns ts, bid, bid_qty, ask, ask_qty, e (record=True). "
          "A missing bid is NO_BID, a missing ask NO_ASK.");

  py::class_<QueuePositions, Listener>(
      am, "QueuePositions",
      "Every resting order's place in its queue, rebuilt from execution reports, with a history "
      "for tracked orders.")
      .def(py::init<>())
      .def("ahead", &QueuePositions::ahead, py::arg("id"),
           "QueuePosition(orders, qty) ahead of the order, or None if it is not resting.")
      .def("track", &QueuePositions::track, py::arg("id"),
           "Record this order's place whenever it changes, from now or from its arrival.")
      .def("flush", &QueuePositions::flush, "Record pending changes now (end of a run).")
      .def("reset", &QueuePositions::reset, "Forget the history.")
      .def_property_readonly("order_count", &QueuePositions::order_count)
      .def(
          "orders",
          [](const QueuePositions& q, Side s) {
            std::vector<std::tuple<OrderId, std::int64_t, Quantity>> out;
            for (const auto& o : q.orders(s)) out.emplace_back(o.id, o.price.ticks, o.leaves);
            return out;
          },
          py::arg("side"), "(id, price, open qty) in priority order, as the mirror sees them.")
      .def(
          "history",
          [](QueuePositions& q, bool flush) {
            if (flush) q.flush();
            const auto& h = q.history();
            const std::size_t n = h.size();
            auto ts = column<std::uint64_t>(n);
            auto id = column<std::uint64_t>(n);
            auto oa = column<std::int64_t>(n);
            auto qa = column<std::int64_t>(n);
            auto lv = column<std::int64_t>(n);
            auto st = column<std::uint8_t>(n);
            for (std::size_t i = 0; i < n; ++i) {
              ts.mutable_data()[i] = h[i].ts;
              id.mutable_data()[i] = h[i].id;
              oa.mutable_data()[i] = static_cast<std::int64_t>(h[i].ahead.orders);
              qa.mutable_data()[i] = h[i].ahead.qty;
              lv.mutable_data()[i] = h[i].leaves;
              st.mutable_data()[i] = static_cast<std::uint8_t>(h[i].status);
            }
            py::dict d;
            d["ts"] = ts;
            d["id"] = id;
            d["orders_ahead"] = oa;
            d["qty_ahead"] = qa;
            d["leaves"] = lv;
            d["status"] = st;
            return d;
          },
          py::arg("flush") = true,
          "Tracked samples as numpy columns ts, id, orders_ahead, qty_ahead, leaves, status "
          "(a QueueStatus value).");
}

}  // namespace
}  // namespace lob::python

PYBIND11_MODULE(_lob, m) {
  using namespace lob;
  using namespace lob::python;
  m.doc() = "C++20 price-time priority order book (quant-stack Project 2).";

  def_enum<Side>(m, "Side", "Order side.", {{"Buy", Side::Buy}, {"Sell", Side::Sell}});
  def_enum<OrderType>(m, "OrderType", "Order types: see order_book.hpp.",
                      {{"Limit", OrderType::Limit},
                       {"Market", OrderType::Market},
                       {"IOC", OrderType::IOC},
                       {"FOK", OrderType::FOK},
                       {"PostOnly", OrderType::PostOnly}});
  def_enum<ExecType>(m, "ExecType", "Execution report types.",
                     {{"New", ExecType::New},
                      {"Fill", ExecType::Fill},
                      {"Cancelled", ExecType::Cancelled},
                      {"Replaced", ExecType::Replaced},
                      {"Rejected", ExecType::Rejected}});
  // C++ Reason::None is Reason.None_ in Python: `None` is a keyword.
  def_enum<Reason>(m, "Reason", "Why an order was rejected, cancelled or expired.",
                   {{"None_", Reason::None},
                    {"DuplicateOrderId", Reason::DuplicateOrderId},
                    {"InvalidQuantity", Reason::InvalidQuantity},
                    {"InvalidPrice", Reason::InvalidPrice},
                    {"UnknownOrder", Reason::UnknownOrder},
                    {"WouldCross", Reason::WouldCross},
                    {"InsufficientLiquidity", Reason::InsufficientLiquidity},
                    {"UserCancel", Reason::UserCancel},
                    {"Unfilled", Reason::Unfilled}});

  def_seq<Trade>(m, "orderbook.Trade", "One maker order matched; price is the maker's price.",
                 {{"taker", "id of the incoming order"},
                  {"maker", "id of the resting order"},
                  {"taker_side", "Side of the incoming order"},
                  {"price", "ticks"},
                  {"qty", "shares traded"},
                  {"ts", "event time, ns"}});
  def_seq<LevelUpdate>(m, "orderbook.LevelUpdate",
                       "New total size and order count of a price level; qty 0 = level gone.",
                       {{"side", "Side"},
                        {"price", "ticks"},
                        {"qty", "total resting size after the change"},
                        {"count", "resting orders after the change"},
                        {"ts", "event time, ns"}});
  def_seq<ExecReport>(m, "orderbook.ExecReport", "One step in an order's life.",
                      {{"id", "order id"},
                       {"type", "ExecType"},
                       {"reason", "Reason (Reason.None_ unless rejected, cancelled or expired)"},
                       {"side", "Side"},
                       {"price", "limit price (New/Replaced/Cancelled) or trade price (Fill)"},
                       {"last_qty", "traded (Fill) or cancelled (Cancelled) by this event"},
                       {"leaves", "still open after this event"},
                       {"ts", "event time, ns"}});
  def_seq<Result>(m, "orderbook.Result",
                  "Outcome of one call. Accepted: filled + resting + cancelled == quantity sent.",
                  {{"accepted", "bool"},
                   {"reason", "Reason"},
                   {"filled", "traded during the call"},
                   {"resting", "left on the book after the call"},
                   {"cancelled", "removed without trading"}});
  def_seq<LevelView>(m, "orderbook.LevelView", "One price level.",
                     {{"price", "ticks"}, {"qty", "total resting size"}, {"count", "orders"}});
  def_seq<OrderView>(m, "orderbook.OrderView", "A resting order; qty is its open quantity.",
                     {{"id", "order id"},
                      {"side", "Side"},
                      {"type", "OrderType"},
                      {"price", "ticks"},
                      {"qty", "open quantity"},
                      {"ts", "time it joined its queue, ns"}});

  bind_book<MapLevels>(m, "OrderBookView", "OrderBook", "std::map price levels");
  bind_book<ArrayLevels>(m, "ArrayOrderBookView", "ArrayOrderBook",
                         "tick-indexed array price levels");

  // ---- LOBSTER ----
  auto lm = m.def_submodule("lobster", "LOBSTER message/orderbook files and replay.");
  using namespace lob::lobster;

  def_enum<lobster::Event>(lm, "Event", "LOBSTER message types (column 2 of the message file).",
                           {{"Submit", lobster::Event::Submit},
                            {"Cancel", lobster::Event::Cancel},
                            {"Delete", lobster::Event::Delete},
                            {"ExecuteVisible", lobster::Event::ExecuteVisible},
                            {"ExecuteHidden", lobster::Event::ExecuteHidden},
                            {"Cross", lobster::Event::Cross},
                            {"Halt", lobster::Event::Halt}});

  py::class_<Message>(lm, "Message", "One row of a LOBSTER message file.")
      .def(py::init([](Timestamp ns, lobster::Event event, OrderId id, Quantity size,
                       std::int64_t price, Side side) {
             return Message{ns, event, id, size, Price{price}, side};
           }),
           py::kw_only(), py::arg("ns"), py::arg("event"), py::arg("id"), py::arg("size"),
           py::arg("price"), py::arg("side"))
      .def_readonly("ns", &Message::ns, "Nanoseconds after midnight.")
      .def_readonly("event", &Message::event)
      .def_readonly("id", &Message::id)
      .def_readonly("size", &Message::size)
      .def_property_readonly("price", [](const Message& msg) { return msg.price.ticks; })
      .def_readonly("side", &Message::side, "Side of the resting order.")
      .def("__repr__", [](const Message& msg) {
        std::ostringstream os;
        os << "Message(ns=" << msg.ns << ", event=Event." << lobster::to_string(msg.event)
           << ", id=" << msg.id << ", size=" << msg.size << ", price=" << msg.price.ticks
           << ", side=Side." << lob::to_string(msg.side) << ")";
        return os.str();
      });

  py::class_<Snapshot>(lm, "Snapshot",
                       "One row of a LOBSTER orderbook file: occupied levels, best first, as "
                       "(price, qty) pairs.")
      .def(py::init([](const Levels& asks, const Levels& bids) {
             return Snapshot{from_pairs(asks), from_pairs(bids)};
           }),
           py::kw_only(), py::arg("asks") = Levels{}, py::arg("bids") = Levels{})
      .def_property_readonly("asks", [](const Snapshot& s) { return to_pairs(s.asks); })
      .def_property_readonly("bids", [](const Snapshot& s) { return to_pairs(s.bids); })
      .def("__eq__",
           [](const Snapshot& a, const Snapshot& b) { return a.asks == b.asks && a.bids == b.bids; },
           py::is_operator())
      .def("__repr__", [](const Snapshot& s) {
        std::ostringstream os;
        os << "Snapshot(asks=" << s.asks.size() << " levels, bids=" << s.bids.size()
           << " levels)";
        return os.str();
      });

  py::class_<ReplayStats>(lm, "ReplayStats", "What a replay saw and how well it matched.")
      .def_readonly("events", &ReplayStats::events)
      .def_readonly("by_type", &ReplayStats::by_type, "Event counts indexed by Event value.")
      .def_readonly("mismatches", &ReplayStats::mismatches)
      .def_readonly("first_mismatch", &ReplayStats::first_mismatch)
      .def_readonly("first_mismatch_detail", &ReplayStats::first_mismatch_detail)
      .def_readonly("errors", &ReplayStats::errors)
      .def_readonly("first_error", &ReplayStats::first_error)
      .def_readonly("first_error_detail", &ReplayStats::first_error_detail)
      .def_readonly("seeded_levels", &ReplayStats::seeded_levels)
      .def_readonly("revealed_levels", &ReplayStats::revealed_levels)
      .def_readonly("revealed_already_right", &ReplayStats::revealed_already_right)
      .def_readonly("unknown_id_events", &ReplayStats::unknown_id_events)
      .def_readonly("executions_matched", &ReplayStats::executions_matched)
      .def_readonly("executions_out_of_order", &ReplayStats::executions_out_of_order)
      .def_readonly("comparisons", &ReplayStats::comparisons)
      .def("__repr__", [](const ReplayStats& s) {
        std::ostringstream os;
        os << "ReplayStats(events=" << s.events << ", mismatches=" << s.mismatches
           << ", errors=" << s.errors << ", comparisons=" << s.comparisons << ")";
        return os.str();
      });

  py::class_<FileInfo>(lm, "FileInfo", "Facts encoded in a LOBSTER file name.")
      .def_readonly("ticker", &FileInfo::ticker)
      .def_readonly("date", &FileInfo::date)
      .def_readonly("levels", &FileInfo::levels)
      .def_readonly("orderbook_path", &FileInfo::orderbook_path);

  lm.def("describe_file", &describe_file, py::arg("message_path"));
  lm.def("parse_message", [](const std::string& line) { return parse_message(line); },
         py::arg("line"));
  lm.def("parse_snapshot",
         [](const std::string& line, std::size_t levels) {
           Snapshot s;
           parse_snapshot(line, levels, s);
           return s;
         },
         py::arg("line"), py::arg("levels"));

  py::class_<PyDayReader>(lm, "DayReader",
                          "Iterates a day as (Message, Snapshot) pairs, both files in lockstep.")
      .def(py::init<const std::string&, const std::optional<std::string>&,
                    std::optional<std::size_t>>(),
           py::arg("message_path"), py::arg("orderbook_path") = py::none(),
           py::arg("levels") = py::none())
      .def("__iter__", [](py::object self) { return self; })
      .def("__next__", &PyDayReader::next)
      .def_property_readonly("rows", &PyDayReader::rows);

  bind_replay<OrderBook>(lm, "Replay", "std::map price levels");
  bind_replay<ArrayOrderBook>(lm, "ArrayReplay", "tick-indexed array price levels");

  auto am = m.def_submodule("analytics", "Microstructure analytics computed from book events.");
  bind_analytics(am);
}

const std = @import("std");

pub fn GenerationalSparseSet(Value: type) type {
    return struct {
        const This = @This();
        pub const DenseIndex = u32;
        pub const SparseIndex = u32;
        pub const Gen = u31;

        sparse_to_dense: std.ArrayList(Sparse),
        dense_to_sparse: std.ArrayList(SparseIndex),
        dense: std.ArrayList(Value),

        pub const empty: This = .{
            .sparse_to_dense = .empty,
            .dense_to_sparse = .empty,
            .dense = .empty,
        };

        pub const Ref = packed struct(u64) {
            slot: SparseIndex,
            gen: Gen,
            padding: u1 = 0,

            comptime {
                std.debug.assert(@bitOffsetOf(Ref, "slot") == 0);
                std.debug.assert(@bitOffsetOf(Ref, "gen") == 32);
            }
        };

        pub const Sparse = packed struct(u64) {
            dense_slot: DenseIndex,
            gen: Gen,
            exists: bool,

            comptime {
                std.debug.assert(@bitOffsetOf(Sparse, "dense_slot") == 0);
                std.debug.assert(@bitOffsetOf(Sparse, "gen") == 32);
                std.debug.assert(@bitOffsetOf(Sparse, "exists") == 63);
            }
        };

        pub fn deinit(set: *This, alloc: std.mem.Allocator) void {
            set.sparse_to_dense.deinit(alloc);
            set.dense_to_sparse.deinit(alloc);
            set.dense.deinit(alloc);
        }

        pub fn append(set: *This, alloc: std.mem.Allocator, value: Value) !Ref {
            const sparse_slot = try set.getFreeSparseIndex(alloc);
            const sparse = &set.sparse_to_dense.items[sparse_slot];
            try set.dense.append(alloc, value);
            try set.dense_to_sparse.append(alloc, sparse_slot);

            sparse.* = .{
                .dense_slot = @intCast(set.dense.items.len - 1),
                .gen = sparse.gen +% 1,
                .exists = true,
            };

            return .{
                .slot = sparse_slot,
                .gen = sparse.gen,
            };
        }

        pub fn insertAtRef(set: *This, alloc: std.mem.Allocator, ref: Ref, value: Value) !void {
            if (ref.slot >= set.sparse_to_dense.items.len) {
                const old_len = set.sparse_to_dense.items.len;
                try set.sparse_to_dense.resize(alloc, ref.slot + 1);
                @memset(set.sparse_to_dense.items[old_len..], .{
                    .dense_slot = 0,
                    .gen = 0,
                    .exists = false,
                });
            }

            const sparse = &set.sparse_to_dense.items[ref.slot];
            if (sparse.exists) return error.AlreadyExists;

            sparse.exists = true;
            sparse.gen = ref.gen;

            try set.dense.append(alloc, value);
            try set.dense_to_sparse.append(alloc, ref.slot);
            sparse.dense_slot = @intCast(set.dense.items.len - 1);
        }

        pub fn swapRemove(set: *This, ref: Ref) error{OutdatedRef}!void {
            const sparse = &set.sparse_to_dense.items[ref.slot];
            if (!sparse.exists or sparse.gen != ref.gen) return error.OutdatedRef;

            const i = sparse.dense_slot;
            const last = set.dense.items.len - 1;

            sparse.exists = false;
            if (i != last) {
                const moved_sparse_index = set.dense_to_sparse.items[last];

                _ = set.dense.swapRemove(i);
                _ = set.dense_to_sparse.swapRemove(i);

                set.sparse_to_dense.items[moved_sparse_index].dense_slot = i;
            } else {
                _ = set.dense.pop();
                _ = set.dense_to_sparse.pop();
            }
        }

        pub fn get(set: *const This, ref: Ref) ?Value {
            if (ref.slot >= set.sparse_to_dense.items.len) return null;

            const sparse = set.sparse_to_dense.items[ref.slot];
            if (!sparse.exists or sparse.gen != ref.gen) return null;

            return set.dense.items[sparse.dense_slot];
        }

        pub fn getPtr(set: *This, ref: Ref) ?*Value {
            if (ref.slot >= set.sparse_to_dense.items.len) return null;

            const sparse = set.sparse_to_dense.items[ref.slot];
            if (!sparse.exists or sparse.gen != ref.gen) return null;

            return &set.dense.items[sparse.dense_slot];
        }

        pub fn refFromDenseIndex(set: *const This, dense_index: DenseIndex) Ref {
            const slot = set.dense_to_sparse.items[dense_index];
            const sparse = set.sparse_to_dense.items[slot];
            return .{
                .slot = sparse.dense_slot,
                .gen = sparse.gen,
            };
        }

        pub fn getFreeSparseIndex(set: *This, alloc: std.mem.Allocator) !SparseIndex {
            for (set.sparse_to_dense.items, 0..) |sparse, i| {
                if (!sparse.exists) return @intCast(i);
            }

            try set.sparse_to_dense.append(alloc, .{
                .dense_slot = 0,
                .gen = 0,
                .exists = false,
            });
            return @intCast(set.sparse_to_dense.items.len - 1);
        }
    };
}

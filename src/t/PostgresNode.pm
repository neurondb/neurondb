package PostgresNode;

use strict;
use warnings;
use Cwd;
use File::Path qw(make_path remove_tree);
use File::Spec;
use IPC::Run qw(run);
use POSIX qw(:sys_wait_h);
use IO::Socket::INET;

=head1 NAME

PostgresNode - PostgreSQL test node management

=head1 SYNOPSIS

  use PostgresNode;

  my $node = PostgresNode->new('test_node');
  $node->init();
  $node->start();
  $node->psql('postgres', 'SELECT 1');
  $node->stop();

=head1 DESCRIPTION

Manages PostgreSQL test instances for TAP testing. Provides methods
for initializing, starting, stopping, and querying PostgreSQL nodes.

=cut

sub new {
	my ($class, $name, %params) = @_;
	my $self = {
		name => $name,
		port => $params{port} || _find_free_port(),
		host => $params{host} || 'localhost',
		data_dir => $params{data_dir} || File::Spec->catfile(
			File::Spec->tmpdir(), "pg_node_${name}_$$"
		),
		pg_config => $params{pg_config} || _find_pg_config(),
		pg_ctl => undef,
		initdb => undef,
		psql => undef,
		pid => undef,
		running => 0,
	};
	bless $self, $class;
	$self->_find_binaries();
	return $self;
}

sub _find_pg_config {
	my $pg_config = $ENV{PG_CONFIG} || 'pg_config';
	my $path = `$pg_config --bindir 2>/dev/null`;
	chomp $path;
	return $path if $path && -d $path;
	
	# Try common locations
	for my $dir (
		'/usr/local/pgsql/bin',
		'/usr/pgsql-*/bin',
		'/usr/lib/postgresql/*/bin',
		'/opt/homebrew/opt/postgresql@*/bin',
	) {
		my @matches = glob($dir);
		for my $match (@matches) {
			return $match if -d $match && -x "$match/pg_ctl";
		}
	}
	die "Cannot find PostgreSQL binaries. Set PG_CONFIG or ensure PostgreSQL is installed.\n";
}

sub _find_binaries {
	my $self = shift;
	my $bindir = $self->{pg_config};
	
	$self->{pg_ctl} = File::Spec->catfile($bindir, 'pg_ctl');
	$self->{initdb} = File::Spec->catfile($bindir, 'initdb');
	$self->{psql} = File::Spec->catfile($bindir, 'psql');
	$self->{postgres} = File::Spec->catfile($bindir, 'postgres');
	
	for my $bin (qw(pg_ctl initdb psql postgres)) {
		my $path = $self->{$bin};
		die "Cannot find $bin at $path\n" unless -x $path;
	}
}

sub _find_free_port {
	my $port = 15432 + int(rand(10000));
	
	# Check if port is available
	my $sock = IO::Socket::INET->new(
		LocalPort => $port,
		Proto => 'tcp',
		Listen => 1,
		Reuse => 1,
	);
	if ($sock) {
		close $sock;
		return $port;
	}
	
	# Fallback: try sequential ports
	for my $i (0..100) {
		my $test_port = 15432 + $i;
		my $test_sock = IO::Socket::INET->new(
			LocalPort => $test_port,
			Proto => 'tcp',
			Listen => 1,
			Reuse => 1,
		);
		if ($test_sock) {
			close $test_sock;
			return $test_port;
		}
	}
	
	die "Cannot find free port\n";
}

=head2 init

Initialize PostgreSQL data directory.

=cut

sub init {
	my ($self, %params) = @_;
	
	return if -d $self->{data_dir} && -f File::Spec->catfile($self->{data_dir}, 'PG_VERSION');
	
	make_path($self->{data_dir}) unless -d $self->{data_dir};
	
	my @cmd = (
		$self->{initdb},
		'-D', $self->{data_dir},
		'--no-locale',
		'--encoding=UTF8',
		'--auth-local=trust',
		'--auth-host=trust',
	);
	
	my ($stdout, $stderr);
	my $result = run \@cmd, '>', \$stdout, '2>', \$stderr;
	
	unless ($result) {
		die "initdb failed: $stderr\n";
	}
	
	# Configure postgresql.conf
	$self->_update_config();
	
	return 1;
}

sub _update_config {
	my $self = shift;
	my $conf_file = File::Spec->catfile($self->{data_dir}, 'postgresql.conf');
	
	# Find neurondb library location
	my $libdir = $self->_find_libdir();
	# Try both .so (Linux) and .dylib (macOS) extensions
	my $neurondb_lib_so = File::Spec->catfile($libdir, 'neurondb.so');
	my $neurondb_lib_dylib = File::Spec->catfile($libdir, 'neurondb.dylib');
	my $neurondb_lib = (-f $neurondb_lib_dylib) ? $neurondb_lib_dylib : $neurondb_lib_so;
	
	open my $fh, '>>', $conf_file or die "Cannot open $conf_file: $!\n";
	print $fh "\n# Test configuration - Optimized for TAP tests\n";
	print $fh "port = $self->{port}\n";
	print $fh "listen_addresses = '$self->{host}'\n";
	
	# Preload neurondb extension
	if (-f $neurondb_lib) {
		print $fh "shared_preload_libraries = 'neurondb'\n";
		print $fh "# NeurondB library: $neurondb_lib\n";
	} else {
		warn "Warning: neurondb.so/neurondb.dylib not found at $libdir, skipping preload\n";
		warn "  Tried: $neurondb_lib_so\n";
		warn "  Tried: $neurondb_lib_dylib\n";
	}
	
	# Optimized settings for testing
	print $fh "max_connections = 20\n";
	print $fh "shared_buffers = 128kB\n";
	print $fh "dynamic_shared_memory_type = posix\n";
	print $fh "max_wal_size = 32MB\n";
	print $fh "min_wal_size = 32MB\n";
	
	# Faster checkpoint for tests
	print $fh "checkpoint_timeout = 5min\n";
	print $fh "checkpoint_completion_target = 0.9\n";
	
	# Disable expensive features for faster startup
	print $fh "autovacuum = off\n";
	print $fh "track_activities = off\n";
	print $fh "track_counts = off\n";
	print $fh "track_io_timing = off\n";
	print $fh "track_functions = none\n";
	
	# Logging
	print $fh "log_line_prefix = '%m [%p] '\n";
	print $fh "log_timezone = 'UTC'\n";
	print $fh "logging_collector = off\n";
	print $fh "log_destination = 'stderr'\n";
	print $fh "log_min_messages = warning\n";
	
	# Locale and timezone
	print $fh "datestyle = 'iso, mdy'\n";
	print $fh "timezone = 'UTC'\n";
	print $fh "lc_messages = 'C'\n";
	print $fh "lc_monetary = 'C'\n";
	print $fh "lc_numeric = 'C'\n";
	print $fh "lc_time = 'C'\n";
	print $fh "default_text_search_config = 'pg_catalog.english'\n";
	
	# Performance optimizations for testing
	print $fh "synchronous_commit = off\n";
	print $fh "fsync = off\n";
	print $fh "full_page_writes = off\n";
	print $fh "wal_level = minimal\n";
	print $fh "max_wal_senders = 0\n";
	
	close $fh;
}

sub _find_libdir {
	my $self = shift;
	
	# Try pg_config first
	if (my $pg_config = $ENV{PG_CONFIG} || 'pg_config') {
		my $libdir = `$pg_config --pkglibdir 2>/dev/null`;
		chomp $libdir;
		return $libdir if $libdir && -d $libdir;
	}
	
	# Try common locations
	my $bindir = $self->{pg_config};
	my $pg_version = `$bindir/pg_config --version 2>/dev/null`;
	chomp $pg_version;
	
	# Extract version number
	$pg_version =~ /(\d+)/;
	my $major_version = $1 || '18';
	
	for my $dir (
		"/usr/local/pgsql.$major_version-pge/lib",
		"/usr/local/pgsql/lib",
		"/usr/pgsql-$major_version/lib",
		"/usr/lib/postgresql/$major_version/lib",
	) {
		return $dir if -d $dir && -f File::Spec->catfile($dir, 'neurondb.so');
	}
	
	# Fallback: try to find from bindir
	if ($bindir) {
		my $parent = File::Spec->catdir(File::Spec->splitdir($bindir));
		my $libdir = File::Spec->catdir($parent, 'lib');
		return $libdir if -d $libdir;
	}
	
	# Last resort: return default
	return '/usr/local/pgsql/lib';
}

=head2 start

Start PostgreSQL instance.

=cut

sub start {
	my ($self, %params) = @_;
	
	return if $self->{running};
	
	$self->init() unless -d $self->{data_dir};
	
	my @cmd = (
		$self->{pg_ctl},
		'start',
		'-D', $self->{data_dir},
		'-l', File::Spec->catfile($self->{data_dir}, 'postgresql.log'),
		'-w',
		'-t', '60',  # Increased timeout for extension loading
	);
	
	my ($stdout, $stderr);
	my $result = run \@cmd, '>', \$stdout, '2>', \$stderr;
	
	unless ($result) {
		die "pg_ctl start failed: $stderr\n";
	}
	
	# Wait for server to be ready
	$self->_wait_for_server();
	
	$self->{running} = 1;
	return 1;
}

sub _wait_for_server {
	my $self = shift;
	my $max_attempts = 60;  # Increased timeout for extension loading
	my $attempt = 0;
	
	while ($attempt < $max_attempts) {
		my $result = $self->psql('postgres', 'SELECT 1', quiet => 1);
		return 1 if $result->{exit_code} == 0;
		sleep 1;
		$attempt++;
	}
	
	# Try to get error from log file
	my $log_file = File::Spec->catfile($self->{data_dir}, 'postgresql.log');
	if (-f $log_file) {
		open my $fh, '<', $log_file or die "Cannot open log file: $!\n";
		my @lines = <$fh>;
		close $fh;
		my $error_msg = join('', grep { /ERROR|FATAL|PANIC/ } @lines[-10..-1]);
		die "PostgreSQL server did not start within $max_attempts seconds\n$error_msg\n";
	}
	
	die "PostgreSQL server did not start within $max_attempts seconds\n";
}

=head2 stop

Stop PostgreSQL instance.

=cut

sub stop {
	my ($self, %params) = @_;
	
	return unless $self->{running};
	
	my @cmd = (
		$self->{pg_ctl},
		'stop',
		'-D', $self->{data_dir},
		'-m', $params{mode} || 'fast',
		'-w',
		'-t', '30',
	);
	
	my ($stdout, $stderr);
	my $result = run \@cmd, '>', \$stdout, '2>', \$stderr;
	
	$self->{running} = 0;
	return 1;
}

=head2 cleanup

Remove data directory and stop instance.

=cut

sub cleanup {
	my $self = shift;
	
	$self->stop() if $self->{running};
	
	if (-d $self->{data_dir}) {
		remove_tree($self->{data_dir}, { error => \my $err });
		if (@$err) {
			warn "Error cleaning up $self->{data_dir}: @$err\n";
		}
	}
}

=head2 psql

Execute SQL query using psql.

  my $result = $node->psql('database', 'SELECT 1');
  print $result->{stdout};

=cut

sub psql {
	my ($self, $dbname, $sql, %params) = @_;
	
	$dbname ||= 'postgres';
	
	# Use current user instead of 'postgres' if not specified
	my $user = $params{user} || $ENV{USER} || getpwuid($<) || 'postgres';
	
	my @cmd = (
		$self->{psql},
		'-h', $self->{host},
		'-p', $self->{port},
		'-U', $user,
		'-d', $dbname,
	);
	
	push @cmd, '-q' if $params{quiet};
	push @cmd, '-t' if $params{tuples_only};
	push @cmd, '-A' if $params{no_align};
	
	my ($stdout, $stderr);
	my $result = run \@cmd, '<', \$sql, '>', \$stdout, '2>', \$stderr;
	
	# Check for SQL errors in stderr (psql may exit 0 even with SQL errors)
	my $has_error = ($stderr && $stderr =~ /^(ERROR|FATAL):/m) ? 1 : 0;
	my $success = $result && !$has_error;
	
	return {
		exit_code => $result ? 0 : $? >> 8,
		stdout => $stdout,
		stderr => $stderr,
		success => $success,
	};
}

=head2 safe_psql

Execute SQL query with error checking. Dies on error.

=cut

sub safe_psql {
	my ($self, $dbname, $sql, %params) = @_;
	
	my $result = $self->psql($dbname, $sql, %params);
	
	unless ($result->{success}) {
		die "psql failed: $result->{stderr}\nSQL: $sql\n";
	}
	
	return $result;
}

=head2 get_port

Get the port number for this node.

=cut

sub get_port {
	my $self = shift;
	return $self->{port};
}

=head2 get_host

Get the host for this node.

=cut

sub get_host {
	my $self = shift;
	return $self->{host};
}

=head2 is_running

Check if node is running.

=cut

sub is_running {
	my $self = shift;
	return $self->{running};
}

=head2 DESTROY

Cleanup on object destruction.

=cut

sub DESTROY {
	my $self = shift;
	$self->cleanup() if $self->{data_dir} && -d $self->{data_dir};
}

1;

